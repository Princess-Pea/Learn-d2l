import torch
from torch import nn
from d2l import torch as d2l

def conv_block(in_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, nums_convs, in_channels, nums_channels):
        super(DenseBlock,self).__init__()
        self.net = nn.Sequential()
        layer=[]
        for i in range(nums_convs):
            layer.append(conv_block(nums_channels*i+in_channels, nums_channels)) 
            # 每个卷积层的输入通道数是初始输入通道数加上前面卷积层输出通道数的总和，因为前面所有卷积层的输出都被连接到这里了
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连接起来
        return X
    # 每次卷积层的输出都与输入连接起来，这样后续的卷积层就可以利用前面所有卷积层的输出特征

blk = DenseBlock(10, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)
# torch.Size([4, 103, 8, 8]) 
# 因为每个卷积层输出10个通道，十个卷积层总共输出100个通道，加上初始输入的3个通道，总共是103个通道。
# 具体来说，第一次卷积层输出10个通道，第二次卷积层输入是初始的3个通道加上第一次卷积层的10个通道，总共13个通道，第二次卷积层输出10个通道，所以总共是3（初始输入） + 10（第一次卷积层输出） + 10（第二次卷积层输出） + ... = 103个通道。

# 过渡层：在两个密集块之间添加过渡层来控制模型复杂度和过拟合。
# 过渡层通常由一个1x1卷积层和一个2x2平均池化层组成，用于减少通道数和空间尺寸。
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

blk = transition_block(103, 10)
print(blk(Y).shape) # torch.Size([4, 10, 4, 4])
# 其中输入通道数是103，输出通道数是10，过渡层的1x1卷积层将通道数从103减少到10。
# 空间尺寸从8x8变为4x4，平均池化层将空间尺寸减半。