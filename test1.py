import numpy as np
import torch
A = torch.tensor([[3, 4, 5],
                  [3, 4, 5]])
B = torch.tensor([[1, 6, 8],
                  [1, 6, 8]])
C = torch.max(A, B)
D = C.max(1)
temp1 = 0


