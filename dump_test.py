import torch

a = torch.Tensor([[[0,0,1],[1,1,1],[0,0,0]],[[0,0,1],[1,1,1],[0,0,0]]])
b = torch.rand((2,4,3,3))

print(b*a.view(2,1,3,3))