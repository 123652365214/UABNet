import torch
from torch import nn

import torch.nn.functional as F

class GatedDynamicDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(1, 2, 3)):
        super(GatedDynamicDilatedConv, self).__init__()
        self.dilations = dilations

        # 多个空洞卷积分支
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=d, padding=d, bias=False)
            for d in dilations
        ])

        # 动态权重生成模块
        self.dynamic_weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(in_channels, len(dilations)),  # 每个分支生成一个权重
            nn.Softmax(dim=1)  # 归一化
        )

        # 门控权重生成模块
        self.gating_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(out_channels, len(dilations)),  # 每个分支生成一个门控权重
            nn.Sigmoid()  # 门控机制，权重范围 [0, 1]
        )

    def forward(self, x):
        batch_size, _, _, _ = x.shape

        # 动态权重生成
        dynamic_weights = self.dynamic_weight_fc(x)  # [B, len(dilations)]

        # 空洞卷积特征提取
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))  # 每个分支的卷积输出
        outputs = torch.stack(outputs, dim=0)  # [len(dilations), B, C, H, W]

        # 动态加权卷积输出
        dynamic_weights = dynamic_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, len(dilations), 1, 1, 1]
        weighted_outputs = (dynamic_weights.permute(1, 0, 2, 3, 4) * outputs).sum(dim=0)  # [B, C, H, W]

        # 门控机制
        gating_weights = self.gating_fc(weighted_outputs)  # [B, len(dilations)]
        gating_weights = gating_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, len(dilations), 1, 1, 1]

        # 门控后的输出
        gated_outputs = (gating_weights.permute(1, 0, 2, 3, 4) * outputs).sum(dim=0)  # [B, C, H, W]

        return gated_outputs




class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=(1,2,3), bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            GatedDynamicDilatedConv(in_channels=inplanes,out_channels=planes,kernel_size=3,dilations=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvBNR2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0,dilation=1, bias=False):
        super(ConvBNR2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBNR2(inplanes, 32, 1),
            ConvBNR2(32, 32, kernel_size,padding=1,stride=stride,dilation=dilation,bias=bias),
            ConvBNR2(32, planes, 1)
        )


    def forward(self, x):

        x2=self.block(x)

        if x2.size()[2:] != x.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=False)

        x=x+x2
        return x



class Decoder_l(nn.Module):
    def __init__(self, hchannel, channel):
        super(Decoder_l, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3,dilation=(1,2,3))
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=(3,5,7))
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=(7,9,11))
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=(11,13,17))

        self.conv1_2 = Conv1x1(channel, channel) #ConvBlock(channel, channel)

        self.conv3_3 = ConvBNR(channel, channel, 3)

        self.block1=ConvBlock(channel//4,channel//4)


    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])

        x1 = self.dconv5_1(xc[1] + x0 + xc[2])

        x2 = self.dconv7_1(xc[2] + x1 + xc[3])

        x3 = self.dconv9_1(xc[3] + x2)
        x33 = self.block1(x3)
        x22 = self.block1(x33+x2)
        x11 = self.block1(x1+x22)
        x00 = self.block1(x0+x11)
        xx = self.conv1_2(torch.cat((x00, x11, x22, x33), dim=1))
        x = self.conv3_3(x + xx)

        return x



# 模型定义
if __name__ == "__main__":
    # 假设低分辨率和高分辨率图像分别为 4 个批次，每个图像具有 64 个通道，尺寸为 32x32
    lf = torch.randn(4, 64, 32, 32)  # 低分辨率输入
    hf = torch.randn(4, 64, 64, 64)  # 高分辨率输入

    # 模型初始化
    model = Decoder_l(hchannel=64, channel=64)

    # 前向传播
    output = model(lf, hf)

    # 输出形状
    print(f"Low resolution input shape: {lf.shape}")
    print(f"High resolution input shape: {hf.shape}")
    print(f"Output shape: {output.shape}")
