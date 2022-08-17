import torch
import torch.nn as nn


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),  # bias为true则说明学习偏差
            nn.InstanceNorm2d(out_channels),  # 归一化
            nn.LeakyReLU(0.2),  # 激活函数
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,  # 卷积核大小
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []  # 一会让里面其实存放了每一次的结果
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))  # 除最后一个外，步长均为2
            in_channels = feature  # 利用一个循环+block简洁的完成了卷积操作
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))    # 生成channels为1的结果
        self.model = nn.Sequential(*layers)     # 传入layers

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))     # 最后要激活一下
