import torch
import torch.nn as nn


# torch.backends.cudnn.benchmark = False
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):  # down：下采样，act：激活，**kwargs字典参数
        super().__init__()
        self.conv = nn.Sequential(  # 卷积块，可以完成下采样卷积或者保持原size卷积
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),  # 标准化
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # identity不会做任何操作
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):  # 残差块，不改变size
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差块儿


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9, ):  # num_features是通道数的一个公约数,num_residuals残差层数
        super(Generator, self).__init__()
        self.initial = nn.Sequential(  # 初始化
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),  # 原地激活
        )
        self.down_blocks = nn.ModuleList(  # 下采样（增加通道数，减小img尺寸
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),

            ]
        )
        self.residual_block = nn.Sequential(  # 残差块儿（不改变大小
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
            # *4是因为之前的各类操作得到的变量channel已经是4
            # 是4*num_featurs了，这里调用了九次残差块儿，进行训练，大小一直不变
        )
        self.up_blocks = nn.ModuleList(  # 上采样block channels减小，img变大
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),

            ]
        )
        self.last = nn.Conv2d(num_features * 1, img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)  # 初始化
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_block(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


'''
观察代码不难发现，在整个生成器的生成过程中，用到的还是简单基础的知识，只是在一些处理选择上比较特殊
代码利用了残差神经网络 和卷积神经网络集合的方式进行训练
def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)

if __name__ == "__main__":
    test()
'''

