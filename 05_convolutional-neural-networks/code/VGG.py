import torch
from torch import nn
from d2l import torch as d2l

def VGG(conv_nums,input,output):
    layers=[]
    for _ in range(conv_nums):
        layers.append(nn.Conve2d(input,output,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        input=output
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


