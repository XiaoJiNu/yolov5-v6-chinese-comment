# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from utils.plots import Annotator, colors, save_one_box
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iouv [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # detections: 16x6
    # labels: 17x5
    # correct: 16x10
    # 每一行用于表示每个预测目标方框和对应标签方框的IOU是否大于等于0.5-0.95的中的元素，比如预测目标和标签IOU=0.6，
    # 则它在correct所在行为[True, True, True, False, False, False, False, False, False, False]
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou: 17x16 每行表示一个标签方框和每个预测方框的iou值
    iou = box_iou(labels[:, 1:], detections[:, :4])

    # ---------- x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])) 注释----------
    temp1 = (iou >= iouv[0])                       # 17x16
    temp2 = (labels[:, 0:1] == detections[:, 5])   # 17x16
    temp21 = labels[:, 0:1]     # 17x1
    temp22 = detections[:, 5]   # 这种索引方式得到的结果从detections的2维变成了1维，元素数量为16
    temp23 = detections[:, 5:]  # detections为16x6的2维tensor，这种索引方式得到的结果依然为16x1的2维tensor
    # temp2由temp21和temp22得到，则17x1和16的一维tensor经过广播的到了17x16的维度。
    # temp2的每一行表示一个标签方框和16个预测方框的各自类别是否相同

    # temp3: 17x16
    # temp3一行表示一个标签方框和所有预测方框进行比较结果，如果一个预测方框和该行表示的标签gt方框的iou大于0.5且他们的类别相同，
    # 认为它们正确预测了，temp3中对应元素为True。
    temp3 = temp1 & temp2
    # x:两个元组，保存了temp3这个2维tensor中标签gt方框和预测方框正确预测的那些元素的索引，分别为索引的x和y值。
    # x中第一个元组保存的索引实质是***正确预测的gt方框在labels中的索引***
    # x中第二个元组保存的索引实质是***正确预测的预测方框在预测结果detections中的索引***
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:  # 如果存在标签和预测方框正确预测了，进行下面的拼接和去重处理
        temp_matches1 = torch.stack(x, 1)  # 8x2，正确预测了的标签和预测方框在labels和detections中的索引
        temp_matches2 = iou[x[0], x[1]]    # 长度为8的一维tensor，表示正确预测了的标签和预测方框的iou
        temp_matches3 = iou[x[0], x[1]][:, None]  # 8x1，将temp_matches2变成8x1的维度
        # matches: 8x3, 将正确预测了的标签和预测方框的在labels和detections中的索引和iou拼接在一起
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        # ----- 去重处理 -----
        # 如果正确预测了的标签和预测方框的数量不止一个，有可能一个gt框对应多个预测框或和一个预测框对应多个gt框
        # 所以要去重
        if x[0].shape[0] > 1:
            # temp1将正确预测了的标签和预测方框的iou进行升序排序，返回了从小到大的索引值
            temp1 = matches[:, 2].argsort()
            # temp2返回了从大到小的索引值，实现降序排列
            temp2 = matches[:, 2].argsort()[::-1]
            # 将正确预测了的标签和预测方框按照iou值进行降序排序
            matches = matches[matches[:, 2].argsort()[::-1]]

            # matches[:, 1]取出了正确预测的预测方框在detections中的索引元素数组
            temp1 = matches[:, 1]
            # 预测方框去重。如果一个预测框匹配到多个gt，则只取第一次出现的预测框(**即和gt框iou最大那个预测框在detections中的索引值**)。
            # 返回这些索引排序后的元组和排序后每个元素在temp1数组中对应的索引
            temp2 = np.unique(matches[:, 1], return_index=True)
            # 得到预测方框去重后元素在temp1数组中的索引，也即在matches中的索引
            temp3 = np.unique(matches[:, 1], return_index=True)[1]
            # 得到temp3后，在matches中提取出预测方框去重后剩下的元素，
            # matches: 8x3
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            # 同理，matches[:, 0]取出正确预测的gt框在labels中的索引组成的数组。
            # 然后去重，对于一个gt框匹配了多个预测框的情况，取出和这些预测框iou最大的那个gt框在labels中索引
            # *** 此时，matches得到了最终正确预测的gt框和预测框在labels和detections中的索引和iou，***
            # *** 并且排除了一个gt框对应多个预测框和一个预测框对应多个gt框的情况***
            # matches: 8x3
            # 第一列为正确预测的gt框在labels中的索引
            # 第二列为正确预测的预测框在detections中的索引
            # 第三列为正确预测的gt和预测框的iou
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        # 得到最终预测正确的预测框的信息
        # 此时，correct中每一行就是预测正确的预测框，并且得到这个预测框和对应gt的iou值和IOU=0.5-0.95的大小比较结果。
        # 大于为True，小于为False
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def draw_labels_and_predict_boxes(
        pred_boxes,
        label_boxes,
        img_with_pred_boxes,
        img_with_label_boxes,
        names):
    """
    传入所有预测方框和标签方框，以及对应的图片，然后在图片上画框
    pred_boxes: 预测方框，(Array[N, 6]), x1, y1, x2, y2, conf, class
    label_boxes: 标签方框，(Array[M, 5]), class, x1, y1, x2, y2
    img_with_pred_boxes: 用于画预测方框的图片
    img_with_label_boxes: 用于画标签方框的图片
    names: 类别名字
    """
    # c = int(cls)  # integer class
    # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    # annotator.box_label(xyxy, label, color=colors(c, True))

    # draw predict results
    annotator_pred = Annotator(img_with_pred_boxes, line_width=1, example=str(names))
    for *xyxy, conf, cls in reversed(pred_boxes):
        c = int(cls)  # integer class
        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        label = f'{names[c]} {conf:.2f}'
        annotator_pred.box_label(xyxy, label, color=colors(c, True))
    img_with_pred_boxes = annotator_pred.result()

    # draw label results
    annotator_label = Annotator(img_with_label_boxes, line_width=1, example=str(names))
    for cls, *xyxy in reversed(label_boxes):
        c = int(cls)  # integer class
        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        label = f'{names[c]}'
        annotator_label.box_label(xyxy, label, color=colors(c, True))
    img_with_label_boxes = annotator_label.result()

    # for testing
    # cv2.imwrite(r'data/fp_fn_imgs/test_pred.jpg', img_with_pred_boxes)
    # cv2.imwrite(r'data/fp_fn_imgs/test_label.jpg', img_with_label_boxes)

    return img_with_pred_boxes, img_with_label_boxes


def save_fp_fn_img(ori_img_path, fp_fn_save_dir, is_fp_or_fn,
                   img_with_pred_boxes, img_with_label_boxes):
    """
    用于保存检测结果中有误检fp或者漏检fn的图片所对于那个的原始图片，预测图片和标签图片
    ori_img_path: 图片原始路径
    fp_fn_save_dir: 保存所有fp,fn所有图片的文件夹根目录。根目录下有fp,fn两个文件夹目录
    is_fp_or_fn: 表示图片是fp还是fn
    img_with_pred_boxes: 画了预测框的图片
    img_with_label_boxes: 画了标签方框的图片
    """
    img_name = os.path.split(ori_img_path)[-1]
    # 保存画了预测框的图片
    img_pred_name = img_name.replace('.jpg', '_pred.jpg')
    img_pred_save_path = os.path.join(fp_fn_save_dir, is_fp_or_fn, img_pred_name)
    cv2.imwrite(img_pred_save_path, img_with_pred_boxes)
    # 保存画了标签方框的图片
    img_label_name = img_name.replace('.jpg', '_label.jpg')
    img_label_save_path = os.path.join(fp_fn_save_dir, is_fp_or_fn, img_label_name)
    cv2.imwrite(img_label_save_path, img_with_label_boxes)
    # 保存原始图片
    img_save_path = os.path.join(fp_fn_save_dir, is_fp_or_fn, img_name)
    shutil.copy(ori_img_path, img_save_path)


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        fp_fn_save_dir=r'data/fp_fn_imgs'
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)

        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres, iou_thres=iou_thres)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    #  -----计算结果指标的的过程-----
    # 1. 每次遍历一个batch图片
    # 2. infer这个batch图片得到infer结果
    # 3. NMS得到NMS后的结果
    # 4. 计算指标(Metrics部分)，步骤如下
    #      遍历这个batch结果的一张图片的结果
    #          提取这张图片对应的标签结果
    #          将这张图片的预测结果映射回原图尺寸(Predictions部分)
    #          计算每个预测狂在IOU=0.5到IOU=0.95时是否预测正确，即是否有满足大于对应IOU指标的标签。结果保存在correct这个tensor中
    #          保存每张图片的(correct, conf, pcls, tcls)结果，conf, pcls, tcls分别表示目标的预测分数、类别以及对应的标签
    #          如果要要是结果，还要计算混淆矩阵。confusion_matrix.process_batch(predn, labelsn)实现。(重点，这里可以得到FP和FN，
    #      用于提取预测错误的结果)

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # one item of out is [x1, y1, x2, y2, conf, cls]
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # ---------- 筛选FP，FN样本 ----------
        # 1. 对于每张图片，在confusion_matrix.process_batch(predn, labelsn)中返回FP，FN的数量。以一个字典的形式返回，
        #    字典形式如下：{"fp_num": fp_count, "fn_num": fn_count}
        # 2. 如果FN > 0，将这个样本保存在FN的文件夹中
        # 3. 如果FN = 0 && FP > 0，将这个样本保存在FP的文件夹中
        # 4. 如果FN = 0 && FP = 0，不筛选这个图片

        # Metrics
        for si, pred in enumerate(out):
            # -----一次处理一张图片-----
            # si是当前图片在这个batch中的id
            # pred是预测值
            # targets中，一个方框包含信息为[batch_id, class_id, x, y, w, h]
            # targets[:, 0] == si取出了标签中对应第si张图的在targets中所有方框标签。然后targets[targets[:, 0] == si, 1:]
            # 取出了这些标签中的[class_id, x, y, w, h]
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]  # 得到当前处理图片的路径和维度h,w
            # correct: [npr x niou] = [npr x 10]，后面用于存储每个预测框在IOU=[0.5:0.95]范围内每个IOU情况下，这个预测框是否有对应真是标签
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # labelsn一个目标为[class_id, x1, y1, x2, y2]
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)

                # ---------- yr: 分别在图上画出标签方框和预测方框----------
                # 1. 复制出两张新的图片
                # 2. 调用annotator.box_label(xyxy, label, color=colors(c, True))分别画标签方框和预测方框
                # 3. 传入save_fp_fn_img()函数保存标签结果和预测结果
                # img_numpy = im.cpu().numpy()[si, ...]
                # img_with_pred_boxes = np.transpose(img_numpy, (1, 2, 0))
                # img_with_pred_boxes = np.ascontiguousarray(img_with_pred_boxes)
                img_ori = cv2.imread(str(path))
                img_with_pred_boxes = img_ori.copy()  # 用于画预测方框的图片
                img_with_label_boxes = img_ori.copy()  # 用于画标签方框的图片
                # 画框
                # predn (Array[N, 6]), x1, y1, x2, y2, conf, class
                # labels (Array[M, 5]), class, x1, y1, x2, y2
                # predn = predn[predn[:, 4] > conf_thres]  #

                img_with_pred_boxes, img_with_label_boxes = draw_labels_and_predict_boxes(
                    predn, labelsn, img_with_pred_boxes, img_with_label_boxes, names)

                # ----- 这里会计算FN，FP以及预测正确的样本 -----
                if plots:
                    fp_fn_dict = {}
                    fp_fn_dict = confusion_matrix.process_batch(predn, labelsn)
                    fp_count = fp_fn_dict["fp_num"]
                    fn_count = fp_fn_dict["fn_num"]

                    # 筛选样本
                    if fn_count > 0:
                        is_fp_or_fn = "fn"
                        save_fp_fn_img(path, fp_fn_save_dir, is_fp_or_fn, img_with_pred_boxes, img_with_label_boxes)
                    elif fn_count == 0 and fp_count > 0:
                        is_fp_or_fn = "fp"
                        save_fp_fn_img(path, fp_fn_save_dir, is_fp_or_fn, img_with_pred_boxes, img_with_label_boxes)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

        callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--fp_fn_save_dir', type=str, default=r'data/fp_fn_imgs',
                        help='fp and fn images save directory')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    # print_args(FILE.stem, vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
