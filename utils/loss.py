# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        # hyp这个参数存放在data/hyps/hyp.scratch-low.yaml中
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        # 这里设置了self.na, self.nc, self.nl, self.anchors属性，这些属性是在yolo.py中的Detect类中初始化的
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # yr: 每个目标的预测之为[x, y, w, h, obj_conf, class1_conf, class2_conf, ..., classn_conf]
                # 问题1 为什么 pxy范围在[-0.5, 1]内？
                # 由于会选取相邻格子的anchor用于预测目标，格子以1为刻度划分。
                # 例：如果gt框的x在[0,0.5)内，它左侧偏离1的格子会用于预测该目标，此时x坐标减去左侧格子左上角坐标，得到的偏差范围为[1, 1.5]
                #    如果gt框的x在[0.5,1)内，它右侧偏离1的格子会用于预测该目标，此时x坐标减去右侧格子左上角坐标，得到的偏差范围为[-0.5, 0]
                #    所以，最终需要预测的cx,cy的偏差范围在[-0.5, 1.5]以内
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                # 问题2 为什么是 [0,4]倍的anchors?
                # 将预测限制为0-4,因为和做target的时候让gt的长宽分别与anchor长宽相除不超过4
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """
        p: a list whose shape is [[16x3x80x80x85], [16x3x40x40x85], [16x3x20x20x85]]
        targets: [204x6]
        """
        na, nt = self.na, targets.shape[0]  # number of anchors 3, targets=204
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ----- 1. 将每个gt box对应上3个anchor id，得到[num_anchor x num_gt x 7]-----
        # ----- 这里的7代表(image_id,class,x,y,w,h,anchor_id) -----
        # temp1 = torch.arange(na, device=targets.device)              # 生成一维的向量，值为[0, 1, 2]
        # temp2 = torch.arange(na, device=targets.device).view(na, 1)  # 将[0, 1, 2]转换成三行一列的二维张量
        # .repeat(1, nt)将[3x1]的[[0] 在第一个维度上复制nt次，即在行方向上复制204次，得到维度为3x204的张量。[[0]，[0]，[0]...[0]
        #                         [1]                                                                [1]，[1]，[1]...[1]
        #                         [2]]                                                               [2]，[2]，[2]...[2]]
        # ai: [3x204], ai应该是anchor id的缩写，表示anchor的序号.
        # ai一列对应一个gt box的3个anchor的id
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # temp1 = targets.repeat(na, 1, 1)  # shape: [204x6] --> [3x204x6]
        # temp2 = ai[:, :, None]  # shape: [3x204x1]
        # targets: [3x204x7], 这里将每个目标都分配给3个anchor，即每个目标方框复制了3份，
        # 然后在(image_id,class,x,y,w,h)后面加上anchor的id，变成(image_id,class,x,y,w,h,anchor_id)
        # 技巧：3x204x7可以将204这个维度去掉，假设只有一个目标，变成3x7，则可以看出来每个目标分配给了所有的anchor，即现在假设了3个anchor
        # 都会负责检测每个目标。而后续的操作就是过滤出真正检测目标的anchor
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # ----- 2. 设置后面筛选anchor时gt box的偏移量，用于后面在gt box附近筛选anchor?? -----
        g = 0.5  # bias
        # ----------*****off用于后面筛选分配有anchor的目标所处的格子和它们相邻的格子*****----------
        # *****off的值为什么是[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1]？？*****
        # 答：对应与后面的offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]和gij = (gxy - offsets).long()，
        # gxy，表示目标的坐标，offsets由off复制再由[j]筛选得到
        # [0, 0]：对于分配有anchor的目标，筛选出目标本身所处的格子，所以对应offsets中那些偏移为[0, 0]，gxy - offsets后还是处于这个格子
        # [1, 0]：gxy中，对于处于格子垂直中心线左边的目标，将他们左边的格子筛选出来用于预测这个目标。gxy - offsets得到的坐标处于左边格子，
        # 再用.long()函数得到左边格子的左上角坐标。
        # 如果为gxy + offsets，则应该为[-1, 0]
        # [0, 1]：gxy中，对于处于格子水平中心线上边的目标，将他们上方的格子筛选出来用于预测这个目标。gxy - offsets得到的坐标处于上边格子，
        # 再用.long()函数得到左边格子的左上角坐标。
        # [-1, 0]：gxy中，对于处于格子垂直中心线右边的目标，将他们右边的格子筛选出来用于预测这个目标。gxy - offsets得到的坐标处于右边格子，
        # 再用.long()函数得到右边格子的左上角坐标。
        # [0, -1]：gxy中，对于处于格子水平中心线下边的目标，将他们上方的格子筛选出来用于预测这个目标。gxy - offsets得到的坐标处于下边格子，
        # 再用.long()函数得到下边格子的左上角坐标。
        # 最终，gij = (gxy - offsets).long()筛选出了分配有anchor的目标所处的格子和这些目标相邻的格子用于预测目标
        # *** 注意：这里有*g操作，最终off = [0, 0],[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5] ***
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # ----- 3. 遍历每个特征层，将每个目标分配到这个特征层对应的anchor. -----
        for i in range(self.nl):  # nl为最后用于检测的特征层数量
            # ----- 3.1 将gt box的x,y,w,h转换到当前特征图尺度下的坐标和宽高-----
            anchors = self.anchors[i]           # 得到当前anchor的尺度
            # temp1 = p[i].shape                # 这里得到了当前特征层的维度值，是一维向量，值为[16, 3, 80, 80, 85]
            # temp2 = torch.tensor(p[i].shape)  # 将temp1转换为tensor [16, 3, 80, 80, 85]，对应[b, c, h, w, classes+5]
            # temp3的值为[80, 80, 80, 80],这里取出了temp2中的w,h,w,h，用于t = targets * gain中将每个目标的归一化坐标x,y,w,h
            # 转换成当前特征层尺度下的x,y,w,h
            # temp3 = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            # gain此时的值为tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
            # 这里的索引为什么用[[3, 2, 3, 2]] ？？
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # ----- 将x,y,w,h转换到当前特征图尺度下的坐标和宽高 -----
            # [3x204x7]*[7] --> [3x204x7]，这里用广播的机制相乘，得到了每个目标在当前特征层尺度下的x,y,w,h
            # 此时每个目标的标签为[image_id, class, x, y, w, h, anchor_id]，x,y,w,h已经是当前特征图尺度下的坐标和宽高
            t = targets * gain  # [3x204x7]*[7] --> [3x204x7]
            # ----- 当有目标标签存在时，进行标签和anchor匹配 -----
            if nt:
                # ----- Matches -----
                # ----- 3.2 匹配achor和gt box，筛选出真正被分配有gt box的anchor对应的真实标签-----
                # temp1 = t[:, :, 4:6]  # shape: [3x204x2]
                # temp1的shape: [3x204x2]，取出所有目标的w,h。为了便于理解，可以假设把204维度去掉得到[3x2]的tensor，
                # 可知每个目标的w,h复制了3次
                # temp2 = anchors[:, None]  # shape: [3x2] --> [3x1x2]，这里在[3x2]维中插入了第二维，得到[3x1x2]的维度
                # r的维度变化，[3x204x2] / [3x1x2] --> [3x204x2]，这里是用广播机制将[3x1x2]变成[3x204x2]，相除以后得到每个
                # 目标w,h与每个anchor的w,h的比值。第三维中的2个元素，第一个为w的比值，第二个元素为h的比值
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio, shape:[3x204x2]
                # temp1 = torch.max(r, 1 / r)  # shape: [3x204x2]
                # 这里的torch.max(r, 1 / r)到底怎么操作的？？？
                # torch.max(r, 1 / r)取出了r和1/r中的对应元素的最大值，维度依然是[3x204x2]。得到每个目标的w与每个anchor的w的
                # 比值r和比值倒数1/r中最大那一个和每个目标的h与每个anchor的h的比值r和比值倒数1/r中最大那一个
                # temp2 = torch.max(r, 1 / r).max(2)[0]  # shape:[3x204]
                # 然后.max(2)[0]取出了第2维度中的最大值，即w,h各自与anchor的w,h的比值r和比值的倒数1/r中最大的值。注意：.max(2)会
                # 取出最大值的对应的索引，max(2)[0]则取出了最大值
                # < self.hyp['anchor_t']，因为调用的yolov5s模型，hyp['anchor_t']在data/hyps/hyp.scratch-low.yaml中为4,
                # 所以< self.hyp['anchor_t']则是取出w,h与对应anchor的w,h的比值和比值的倒数都小于4的anchor筛选出来，实际是
                # 比值都处于[1/4,4]之间的目标才会匹配上anchor，如果一个目标的w,h与对应anchor的w,h的比值不在[1/4,4]之间，则
                # 这个目标不会被分配anchor,在j中对应的元素为False
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, shape: [3x204]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # ----------筛选出真正被分配有目标的anchor对应的真实标签----------
                # 从[3x204x7]中过滤出j(维度为[3x204])中为True的值，这些筛选出来的值表示真正被分配有目标的anchor对应的真实标签.
                # 注意，前面targets是被复制为了3份的，所以这里原始的204个目标此时得到了274个标签，因为一个目标的w,h可能跟多个anchor
                # 的w,h比值在[1/4,4]之间
                # 另外，此时标签的x,y,w,h已经被转换到当前特征层尺度坐标系
                t = t[j]  # filter, shape: [3x204x7] --> [274x7]

                # ----- 3.3 计算Offsets ？？ -----
                # 得到所有被分配有目标的anchor对应目标在当前特征层中的真实x, y坐标，即相对于特征层左上角的x,y值
                gxy = t[:, 2:4]  # grid xy, shape: [274x2],
                # 以当前特征层的右下角为原点，得到所有被分配有目标的anchor对应目标在当前特征层中距离最右下角的x,y坐标
                gxi = gain[[2, 3]] - gxy  # inverse, shape: [274x2]
                # g在前面设置为0.5
                # gxy % 1 < g，x坐标小于0.5和y小于0,5
                # temp1 = (gxy % 1 < g)                  # shape: [274x2]
                # temp2 = (gxy > 1)                      # shape: [274x2]
                # temp3 = ((gxy % 1 < g) & (gxy > 1))    # shape: [274x2]
                # temp4 = ((gxy % 1 < g) & (gxy > 1)).T  # shape: [2X274]

                # 由于gxy一行有x,y两个元素，经过((gxy % 1 < g) & (gxy > 1)).T 到了[2x274]的tensor, 拆开得到j,k，为便于理解，我们分开来看。
                # *****先看j*****,
                # 条件j1：(gxy % 1 < g)，在前面分配有目标的anchor对应的真实标签中，筛选出x坐标在所处格子的垂直中心线左边的那些标签，
                # 即gxy中x坐标对1取余数后小于0.5的元素为True，(gxy % 1 < g)实现这个操作
                # 条件j2：(gxy > 1))，在前面分配有目标的anchor对应的真实标签中，筛选出x坐标大于1的那些标签，即gxy中x坐标大于1的元素
                # 为True，(gxy > 1)实现。因为后面处于左边的目标的x坐标会减去1筛选出左边的格子，为防止小于1的情况，所以要x坐标大于1。
                # 筛选出同时满足条件j1和条件j2的标签，即以左上角为原点，筛选出目标的x坐标大于1，且在目标所处的格子垂直中心线左边的那些标签。
                # 这里有274个anchor被分配了标签，满足条件1和条件2和那些标签在j中对应元素为True
                # *****先看k*****,
                # 条件k1：在前面分配有目标的anchor对应的真实标签中，筛选出y坐标在所处格子的水平中心线上边的那些标签，即gxy中y坐标对1取余数后
                # 小于0.5的元素为True，(gxy % 1 < g)实现这个操作
                # 条件k2：在前面分配有目标的anchor对应的真实标签中，筛选出y坐标大于1的那些标签，即gxy中y坐标大于1的元素为True，(gxy > 1)实现
                # 筛选出同时满足条件k1和条件k2的标签，即以左上角为原点，筛选出目标的y坐标大于1，且在目标所处的格子水平中心线上边的那些标签。
                # 因为后面处于格子上边的目标的y坐标会减去1筛选出相邻上方的格子，为防止小于1的情况，所以要y坐标大于1。
                # 这里有274个anchor被分配了标签，满足条件k1和条件k2和那些标签在k中对应元素为True
                j, k = ((gxy % 1 < g) & (gxy > 1)).T     # j: [274], k: [274]
                # ----------对于l,m，此时是找到相对于特征层右下角的x,y坐标满足相关条件的目标----------
                # *****先看l*****
                # 和j一样，l是目标距离当前特征层大于一个格子(gxi > 1)且在目标所所处格子的垂直中心线右边的那些标签(gxi % 1 < g)
                # *****先看m*****
                # 和k一样，m是目标距离当前特征层大于一个格子(gxi > 1)且在目标所所处格子的水平中心线下边的那些标签(gxi % 1 < g)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T     # l: [274], m: [274]
                # *****j,k,l,m总结*****
                # 左边：j，筛选格子垂直中心线左边的目标，这些目标所处格子不包含第一列格子，因为第一列格子左边没有格子，gij = (gxy - offsets).long()中会越界
                # 上边：k，筛选格子水平中信息上边的目标，这些目标所处格子不包含第一行格子，因为第一行格子上边没有格子，gij = (gxy - offsets).long()中会越界
                # 右边：l，筛选格子垂直中心线右边的目标，这些目标所处格子不包含最后一列格子，因为最后一列格子右边没有格子，gij = (gxy - offsets).long()中会越界
                # 下边：m，筛选格子水平中心线下边的目标，这些目标所处格子不包含最后一行格子，因为最后一行格子下边没有格子，gij = (gxy - offsets).long()中会越界
                # 生成的j用于后面t = t.repeat((5, 1, 1))[j]筛选满足每个目标x,y本身，以及在格子中偏左、上、右、下的目标
                # *****相邻格子如何找到的？*****
                # 答：参考off用于后面筛选分配有anchor的目标所处的格子和它们相邻的格子的注释
                j = torch.stack((torch.ones_like(j), j, k, l, m))          # shape: [5x274]

                # -----增加正样本数量-----
                # temp1 = t.repeat((5, 1, 1))                              # shape: [5x274x7]
                # t.repeat((5, 1, 1))[j]筛选出了所有分配有anchor的目标本身，以及这些目标中满足处于格子中偏左、上、右、下条件的目标，
                # 此时正样本数量增加了统一放在了一个tensor中。和后面的offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # 对应起来看，筛选出来的目标的偏移量跟offsets中元素一一对应
                # ***除了特殊的处于边界的目标，一般的目标所处的格子在x,y方向上相邻的格子也会来预测这个目标，相当于正样本数量变为了原来的3倍***
                # ***以水平和垂直中心线将格子划分为4个小格子，如果一个处于左上角的小格子，目标的左边格子和上边格子也会来预测这个目标***
                t = t.repeat((5, 1, 1))[j]                                 # shape: [5x274x7] --> [813x7]

                # -----注释offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]-----
                # temp1 = torch.zeros_like(gxy)[None]   # [274x2] --> [1x274x2]
                # temp2 = off[:, None]                  # [5x2]   --> [5x1x2]
                # temp2_repeat = temp2.repeat(1, gxy.shape[0], 1)  # [5x274x2] gxy.shape[0]得到了目标的数量
                # temp3 = temp1 + temp2                 # [5x274x2]，
                # temp4 = (temp3 - temp2_repeat).sum()  # temp4 = 0
                # temp3 = temp1+temp2 利用了广播原则，temp1和temp2中维度为1的维度复制为对方对应的维度。temp1会将274个目标都生成[0,0]
                # 的值，temp2表示一个目标本身和它左、上、右、下的偏移量。这里temp2会复制274份，给每个目标都分配一个目标本身和它上下左右的偏移量。
                # 为什么不直接用off[:, None].repeat(1, gxy.shape[0], 1)得到temp3 ??
                # 由调试可知temp4=0，所以off[:, None].repeat(1, gxy.shape[0], 1)可以替换(torch.zeros_like(gxy)[None] + off[:, None])
                # 这种操作比(torch.zeros_like(gxy)[None] + off[:, None])更好理解

                # 下面的[j]操作在干什么？
                # temp3 = (torch.zeros_like(gxy)[None] + off[:, None])维度为[5x274x2]，j的维度为[5x274]。这里先不要看[5x274x2]
                # 中x2那个维度，就看j这个[5x274]去筛选temp3中[5x274]的维度。对于j这个5行274列的tensor，0-4行的元素，为True的那些元素分别
                # 表示每个是否为目标x,y本身，以及在格子中偏左、上、右、下的目标。
                # 第一行全部元素为True，筛选除了所有分配有anchor的目标本身的偏移量，为0
                # 第二行中为True的元素筛选出了处于格子左边的目标的x方向偏移量，为0.5
                # 第三行中为True的元素筛选出了处于格子上边的目标的y方向偏移量，为0.5
                # 第四行中为True的元素筛选出了处于格子右边的目标的x方向偏移量，为-0.5
                # 第五行中为True的元素筛选出了处于格子下边的目标的y方向偏移量，为-0.5

                # *****offsets是什么？*****
                # 最终，offsets表示所有分配有anchor的目标，以及这些目标中处于格子中偏左、上、右、下的目标所对应的，后面用于筛选相邻格子
                # 的偏移量。并且和前面的t = t.repeat((5, 1, 1))[j]中[j]筛选出来的目标对一一对应，维度都是[813x2]。由于将目标本身和
                # 这些目标满足在格子中偏左、上、右、下的条件的目标放在了一个tensor中，所以带来了理解上的困难
                # *****如何用offsets用于干什么？*****
                # 在后续的gij = (gxy - offsets).long()中，目标的坐标分别减去offsets中对应的偏移量再取整数，计算出了预测每个目标的
                # 格子(包含本身所处格和相邻的格子)的左上角坐标
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # offsets: [813x2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # ----- 3.4 定义经过匹配增加后的所有gt boxes在当前feature map上的各种参数 -----
            # ----- 包括：所有gt boxes在这个batch中的id、类别、在feature map上的坐标gxy、宽高gwh、所处格子左上角坐标gij-----
            b, c = t[:, :2].long().T  # image, class. b:[813], c:[813]，得到了所有筛选出来的目标在这个batch中的id和类别
            gxy = t[:, 2:4]  # [813x2], grid xy，得到了所有筛选出来的目标的x,y坐标
            gwh = t[:, 4:6]  # [813x2], grid wh，得到了所有筛选出来的目标的w,h值
            # gij是什么？ 由注释offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]可知，计算出了预测每个目标的格子(包含
            # 本身所处格和相邻的格子)的左上角坐标
            gij = (gxy - offsets).long()  # [813x2]
            # gi: 筛选出来的格子的左上角x坐标
            # gj: 筛选出来的格子的左上角y坐标
            gi, gj = gij.T  # grid xy indices, gi:[813], gj:[813x2]

            # Append
            # ----- 3.5 将最终的生成的标签参数添加到各自的列表中，包括以下部分 -----
            # indices: 匹配增加后的所有gt boxes对应的anchor在当前特征图中的索引，用于定位每个gt box在预测值中对应位置
            # tbox: 匹配增加后的所有gt boxes的cx,cy对应的归一化的预测偏差目标值和未归一化的w,h值
            # tcls：匹配增加后的所有gt boxes的类别
            # anch：匹配增加后的所有gt boxes的对应anchor的w,h值

            # a = t[:, 6].long()得到了前面筛选出来的所有分配有anchor的目标本身，以及这些目标中满足处于格子中偏左、上、右、下条件的目标，
            # 跟它们匹配上的anchor的id
            a = t[:, 6].long()  # anchor indices，[813]
            # *****indices用于干什么??*****
            # 答：添加当前特征层中用于定位每个目标匹配的anchor的具体位置
            # 如何定位？如下:
            # gain[2]，gain[3]：从前面的gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]可知，gain[2],gain[3]是当前特征层的宽高
            # gj.clamp_(0, gain[3] - 1):将筛选出来的格子的左上角y坐标限制在当前特征层的高度范围内
            # gi.clamp_(0, gain[2] - 1):将筛选出来的格子的左上角x坐标限制在当前特征层的宽度范围内
            # 最终，indices得到了所有筛选出来的目标对应的图片在这个batch中的id(b)，于它匹配的anchor的id(a),以及用于预测每个目标对应的格子
            # 的左上角坐标x,y。这样就能找到每个目标匹配的anchor的具体位置
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices

            # ----- 3.6 添加当前特征层要预测的x,y,w,h对应的真实标签-----
            # gxy - gij：筛选出来的目标在当前特征图尺度下的坐标减去用于预测它的格子的左上角坐标，得到cx,cy对应的预测偏差目标值
            # gwh:筛选出来的目标在当前特征图尺度下的宽高。
            # *** 归一化的预测偏差目标值在此处实现 ***
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # 添加筛选出来的目标所匹配的anchor的宽高值
            anch.append(anchors[a])  # anchors
            # 添加当前特征层所有筛选出来的目标的类别
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
