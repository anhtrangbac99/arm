import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms 
i = np.load('img.np.npy')
l = np.load('l.np.npy')
t = np.load('t.np.npy')
img = Image.open('GOPRO/train/GOPR0385_11_00/sharp/000184.png')
img = transforms.ToTensor()(img.resize((64,64)))

blur = Image.open('GOPRO/train/GOPR0385_11_00/blur/000184.png')
blur = transforms.ToTensor()(blur.resize((64,64)))

k = torch.Tensor(l[0])#.view(19,19,64,64)#.repeat(3,1,1,1,1)
target = torch.Tensor(t[0])

# print(i.shape)
# print(img.shape)
# print(k.shape)
# print(k)

# torch.nn.functional.conv2d(img,k,stride=0)
C,H,W = img.shape
out = torch.zeros(img.shape)
# for i,ch in enumerate(img):
for i in range(H):
    for j in range(W):
        kernel = k[:,i,j]
        # kernel = kernel.view(19,19)
        kernel = kernel.view( 1,1, 19, 19).repeat( 3,3, 1, 1)
        o = F.conv2d(img.view(1,3,64,64),kernel,padding=9).view(3,64,64)[:,i,j]
        # print(o.shape)
        out[:,i,j] = o
        # print(kernel.shape)
        # out[:][j][k] =

out1 = torch.zeros(img.shape)

for i in range(H):
    for j in range(W):
        kernel = target[:,i,j]
        # kernel = kernel.view(19,19)
        kernel = kernel.view( 1,1, 19, 19).repeat( 3,3, 1, 1)
        o = F.conv2d(img.view(1,3,64,64),kernel,padding=9).view(3,64,64)[:,i,j]
        # print(o.shape)
        out1[:,i,j] = o
        # print(kernel.shape)
        # out[:][j][k] =
out = transforms.ToPILImage()(out)
out.save('out2.png')

blur = transforms.ToPILImage()(blur)
blur.save('blur.png')

out1 = transforms.ToPILImage()(out1)
out1.save('out3.png')
print(out)
print(blur)