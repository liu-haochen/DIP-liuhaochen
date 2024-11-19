import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(3, 64, stride=2)  # 输入通道为3，输出通道为64
        self.encoder2 = self.conv_block(64, 128, stride=2)
        self.encoder3 = self.conv_block(128, 256, stride=2)
        self.encoder4 = self.conv_block(256, 512, stride=2)
        self.encoder5 = self.conv_block(512, 512, stride=2)
        self.encoder6 = self.conv_block(512, 512, stride=2)
        self.encoder7 = self.conv_block(512, 512, stride=2)

        # 解码器部分
        self.decoder7 = self.upconv_block(512, 512)
        self.decoder6 = self.upconv_block(1024, 512)
        self.decoder5 = self.upconv_block(1024, 512)
        self.decoder4 = self.upconv_block(1024, 256)
        self.decoder3 = self.upconv_block(512, 128)
        self.decoder2 = self.upconv_block(256, 64)
        self.decoder1 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def conv_block(self, in_channels, out_channels, stride=2):
        """标准卷积块带有ReLU和批量归一化"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """上采样卷积块带有ReLU和批量归一化"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)

        # 解码器部分
        d7 = self.decoder7(e7)
        d6 = self.decoder6(torch.cat([d7, e6], 1))
        d5 = self.decoder5(torch.cat([d6, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(torch.cat([d2, e1], 1))

        return torch.tanh(d1)  # 输出范围[−1,1]，使用tanh激活函数


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 判别器网络
        self.model = nn.Sequential(
            self.conv_block(6, 64, stride=2),  # 输入是拼接后的图像（生成图像和真实图像拼接）
            self.conv_block(64, 128, stride=2),
            self.conv_block(128, 256, stride=2),
            self.conv_block(256, 512, stride=2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 输出为1通道，表示每个patch是否真实
        )

    def conv_block(self, in_channels, out_channels, stride=2):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        # 拼接输入图像和目标图像
        combined = torch.cat([x, y], 1)  # 拼接方式 [batch, 6, height, width]
        return self.model(combined)
