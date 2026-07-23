import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DWTConv(nn.Module):
    def __init__(self, in_channels, out_channels, levels=1):
        super(DWTConv, self).__init__()
        self.levels = levels
        self.out_channels = out_channels

        # 1x1卷积用于融合小波特征
        self.conv1x1 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
        self.conv_final = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

        # Haar wavelet filters
        self.register_buffer('fLL', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2)
        self.register_buffer('fLH', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('fHL', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('fHH', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2)
        self.relu = nn.ReLU(inplace=False)
    def dwt(self, x):
        """单级小波分解"""
        B, C, H, W = x.shape
        filters = torch.stack([self.fLL, self.fLH, self.fHL, self.fHH], dim=0).unsqueeze(1).repeat(C, 1, 1, 1)
        filters = filters.to(x.device)
        return F.conv2d(x, filters, stride=2, groups=C)

    def iwt(self, x):
        """逆小波变换"""
        B, C, H, W = x.shape
        filters = torch.stack([self.fLL, self.fLH, self.fHL, self.fHH], dim=0).unsqueeze(1).repeat(C // 4, 1, 1, 1)
        filters = filters.to(x.device)
        return F.conv_transpose2d(x, filters, stride=2, groups=C // 4)


    def forward(self, x):
        # 单级小波分解
        x = self.dwt(x)  # 输出通道变为 in_channels*4

        # 1x1卷积融合特征
        x = self.conv1x1(x)  # 通道数变为 out_channels
        # x = F.relu(x)
        x = self.relu(x)

        # 逆变换
        x = self.iwt(x)  # 保持 out_channels

        # 最终1x1卷积
        return self.conv_final(x)


class SPF(nn.Module):
    """Spectral Pooling Filter"""

    def __init__(self, lambda_low=0.7):
        super(SPF, self).__init__()
        self.lambda_low = lambda_low

    def dynamic_pad_to_power_of_two(self,x):
        """
        动态填充输入张量到最近的2的幂次方
        规则：当距离两个2的幂次方距离相等时，选择更大的那个
        输入：x (Tensor) - 形状为[B, C, H, W]
        返回：填充后的Tensor和填充信息(pad_h, pad_w)
        """
        _, _, h, w = x.shape

        # 计算高度方向的目标尺寸
        lower_h = 1 << (math.floor(math.log2(h))) if h > 1 else 1
        upper_h = 1 << (math.ceil(math.log2(h))) if h > 1 else 1
        target_h = upper_h if (h - lower_h) >= (upper_h - h) else lower_h

        # 计算宽度方向的目标尺寸
        lower_w = 1 << (math.floor(math.log2(w))) if w > 1 else 1
        upper_w = 1 << (math.ceil(math.log2(w))) if w > 1 else 1
        target_w = upper_w if (w - lower_w) >= (upper_w - w) else lower_w

        # 计算需要填充的像素数
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # 对称填充（左右/上下各分一半）
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # 执行填充
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return x_padded


    def forward(self, z):
        B, C, H, W = z.shape
        # 计算需要填充的大小（填充到 32x32）
        # pad_h = 32 - z.size(2)  # 32 - 24 = 8
        # pad_w = 32 - z.size(3)  # 32 - 24 = 8
        # # 对称填充（也可以选择零填充）
        # z = F.pad(z, (0, pad_w, 0, pad_h), mode='constant', value=0)  # 现在形状是 [B, C, 32, 32]

        z = self.dynamic_pad_to_power_of_two(z)
        Z = torch.fft.fft2(z)
        Z_shifted = torch.fft.fftshift(Z, dim=(-2, -1))


        lf_radius = min(H, W) // 4
        center_h, center_w = H // 2, W // 2
        mask = torch.zeros_like(Z_shifted.real)
        mask[:, :, center_h - lf_radius:center_h + lf_radius, center_w - lf_radius:center_w + lf_radius] = 1
        S_lf = Z_shifted * mask
        S_hf = Z_shifted * (1 - mask)
        f_lp = torch.fft.ifft2(torch.fft.ifftshift(S_lf, dim=(-2, -1))).real
        f_hp = torch.fft.ifft2(torch.fft.ifftshift(S_hf, dim=(-2, -1))).real
        output = self.lambda_low * f_lp + (1 - self.lambda_low) * f_hp
        return output



class WSPM(nn.Module):
    """Wavelet-based Spectral Pooling Module"""

    def __init__(self, in_channels, out_channels, levels=1, lambda_low=0.7):
        super(WSPM, self).__init__()
        assert out_channels % 2 == 0, "out_channels must be even"
        self.dwt_conv = DWTConv(in_channels, out_channels, levels)
        self.spf1 = SPF(lambda_low)
        self.spf2 = SPF(lambda_low)

        # Depthwise卷积
        self.dw_1xn = nn.Conv2d(out_channels // 2, out_channels // 2,
                                kernel_size=(1, 5), padding=(0, 2), groups=out_channels // 2)
        self.dw_nx1 = nn.Conv2d(out_channels // 2, out_channels // 2,
                                kernel_size=(5, 1), padding=(2, 0), groups=out_channels // 2)

        self.conv_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # DWTConv处理
        _, _, H, W = x.shape
        x = self.dwt_conv(x)  # [B, out_channels, H, W]

        # 通道拆分
        x1, x2 = torch.chunk(x, 2, dim=1)  # 各[B, out_channels//2, H, W]

        # 谱池化滤波
        x1 = self.spf1(x1)
        x2 = self.spf2(x2)

        # 深度可分离卷积
        x1 = self.dw_1xn(x1)
        x2 = self.dw_nx1(x2)
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=True)

        # 合并并输出
        x = torch.cat([x1, x2], dim=1)
        return self.conv_final(x)


class FE_RFB(nn.Module):
    def __init__(self, in_channels, out_channels, n_list=[1,3, 5, 7], rates=[1, 3, 5, 7], lambda_low=0.7):
        """
        Frequency-Enhanced Receptive Field Block
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n_list: WSPM的低频区域半径参数列表
        :param rates: 空洞卷积的dilation rate列表
        :param lambda_low: SPF模块的低频权重
        """
        super(FE_RFB, self).__init__()
        self.out_channels = out_channels

        # 初始激活和3x3卷积 (结构图中的"Activate Compat + Conv 3x3")
        self.activate_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

        # 多分支WSPM (结构图中的WSPM n=3,5,7)
        self.wspm_branches = nn.ModuleList([
            WSPM(out_channels, out_channels, levels=1, lambda_low=lambda_low)
            for _ in n_list
        ])

        # 多尺度空洞卷积 (结构图中的Conv 3x3 rate=1,3,5,7)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=rate, dilation=rate)
            for rate in rates
        ])

        # 最后的1x1卷积 (结构图中的4个Conv 1x1)
        self.final_convs = nn.Sequential(
            nn.Conv2d(out_channels * (len(n_list) + len(rates)), out_channels, kernel_size=1),
                      nn.Conv2d(out_channels, out_channels, kernel_size=1),
                      nn.Conv2d(out_channels, out_channels, kernel_size=1),
                      nn.Conv2d(out_channels, out_channels, kernel_size=1)
                      )

    def forward(self, x):
        # 初始激活和卷积
        x = self.activate_conv(x)

        # WSPM分支处理
        wspm_outputs = []
        for wspm in self.wspm_branches:
            wspm_outputs.append(wspm(x))

        # 空洞卷积分支处理
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))

        # 合并所有分支
        x = torch.cat(wspm_outputs + dilated_outputs, dim=1)

        # 最终1x1卷积序列
        return self.final_convs(x)


if __name__ == '__main__':
    # 测试用例
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # wspm = WSPM(in_channels=64, out_channels=64, levels=1).to(device)
    #
    # # 输入输出测试
    # x = torch.randn(4, 64, 128, 128).to(device)
    # output = wspm(x)
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {output.shape}")
    # 初始化FE-RFB模块
    fe_rfb = FE_RFB(in_channels=64, out_channels=64,
                    n_list=[3, 5, 7], rates=[1, 3, 5, 7]).cuda()

    # 输入测试
    x = torch.randn(4, 64, 24, 24).cuda()
    output = fe_rfb(x)  # 输出形状: [4, 64, 128, 128]

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
