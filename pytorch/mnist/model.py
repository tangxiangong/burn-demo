import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """
    卷积神经网络用于MNIST手写数字识别
    结构：
    - conv1: 卷积层 (3x3)
    - conv2: 卷积层 (3x3)
    - pool: 自适应平均池化 (8x8)
    - dropout: Dropout层 (p=0.5)
    - linear1: 全连接层 (1024->512)
    - linear2: 全连接层 (512->10)
    - activation: ReLU
    """

    def __init__(self):
        super(MNISTConvNet, self).__init__()
        # 第一个卷积层，输入通道为1（灰度图像），输出通道为8，卷积核大小为3x3
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0
        )

        # 第二个卷积层，输入通道为8，输出通道为16，卷积核大小为3x3
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0
        )

        # 自适应平均池化层，输出大小为8x8
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        # Dropout层，概率为0.5
        self.dropout = nn.Dropout(p=0.5)

        # 第一个全连接层，输入维度为16*8*8=1024，输出维度为512
        self.linear1 = nn.Linear(in_features=16 * 8 * 8, out_features=512)

        # 第二个全连接层，输入维度为512，输出维度为10（对应10个数字类别）
        self.linear2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用第一个卷积层和ReLU激活函数
        x = F.relu(self.conv1(x))

        # 应用第二个卷积层和ReLU激活函数
        x = F.relu(self.conv2(x))

        # 应用自适应平均池化
        x = self.pool(x)

        # 展平张量以便输入全连接层
        x = x.view(-1, 16 * 8 * 8)

        # 应用Dropout
        x = self.dropout(x)

        # 应用第一个全连接层和ReLU激活函数
        x = F.relu(self.linear1(x))

        # 应用第二个全连接层（不应用激活函数，因为后续会使用交叉熵损失函数）
        x = self.linear2(x)

        return x
