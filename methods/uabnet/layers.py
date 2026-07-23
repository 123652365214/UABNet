import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sympy.categories.baseclasses import Class
from mmcv.cnn import build_norm_layer
import numpy as np
import cv2
from torch.nn import Parameter, Softmax

from .ops import ConvBNReLU, resize_to
from .units import ConvBNR


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        """A simple ASPP variant.

        Args:
            in_dim (int): Input channels.
            out_dim (int): Output channels.
            dilation (int, optional): Dilation of the convolution operation. Defaults to 3.
        """
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x

class OAA1(nn.Module):
    def __init__(self,cur_in_channels=64,low_in_channels=32,out_channels=16,cur_scale=2,low_scale=1):
        super(OAA1,self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

    def forward(self,x_cur,x_low):
        x_cur = self.cur_conv(x_cur)
        #bicubic bilinear nearest
        x_cur = F.interpolate(x_cur, scale_factor=self.cur_scale,  mode='bicubic',align_corners = False)

        x_low = self.low_conv(x_low)
        x_low = F.interpolate(x_low, scale_factor=self.low_scale,  mode='bicubic',align_corners = False)
        x = torch.cat((x_cur,x_low),dim=1)
        x = self.out_conv(x)
        return x



# def get_open_map(input,kernel_size,iterations):
#     kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
#     open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
#     open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
#     return open_map_tensor.unsqueeze(1).cuda()

def get_open_map(input, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # 0. 维度检查与统一
    original_dim = input.dim()
    if original_dim == 3:  # 如果是[C,H,W]格式
        input = input.unsqueeze(0)  # 升维为[B,C,H,W]

    open_map_list = []
    for i in input.cpu():
        # 1. 处理不同通道情况
        if i.size(0) == 1:  # 单通道 [1,H,W]
            img_np = i.squeeze(0).detach().numpy()  # [H,W]
        else:  # 多通道 [C,H,W]
            img_np = i.permute(1, 2, 0).detach().numpy()  # [H,W,C]

        # 2. 数据类型转换（保持原有方案）
        img_np = _convert_dtype(img_np)

        # 3. 膨胀操作
        dilated = cv2.dilate(img_np, kernel=kernel, iterations=iterations)
        open_map_list.append(dilated)

    # 4. 智能维度恢复
    result = torch.from_numpy(np.array(open_map_list))

    if original_dim == 3:  # 原输入是3D
        return result.squeeze(0).cuda()  # 降维为[C,H,W]
    else:  # 原输入是4D
        if result.dim() == 3:  # 单通道情况
            return result.unsqueeze(1).cuda()  # [B,1,H,W]
        else:
            return result.permute(0, 3, 1, 2).cuda()  # [B,H,W,C] -> [B,C,H,W]


def _convert_dtype(img_np):
    """统一数据类型转换逻辑"""
    if img_np.dtype == np.float64:
        return img_np.astype(np.float32)
    elif img_np.dtype == np.float32 and img_np.max() <= 1.0:
        return (img_np * 255).astype(np.uint8)
    else:
        return img_np.astype(np.uint8)


class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FGC(nn.Module):
    def __init__(self, channel1, channel2,focus_background = True, opr_kernel_size = 3,iterations = 1):
        super(FGC, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations


    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        dep_x = self.up(dep_x)

        input_map = self.input_map(in_map)

        if self.focus_background:
            self.increase_map = self.increase_input_map(get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map)
            b_feature = cur_x * self.increase_map #当前层中,关注深层部分没有关注的部分

        else:
            b_feature = cur_x * input_map  #在当前层中，对深层部分关注的部分更加关注，同时也关注一下其他部分
        #b_feature = cur_x
        fn = self.conv2(b_feature)


        refine2 = self.conv_cur_dep1(torch.cat((dep_x, self.beta * fn),dim=1))
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class FM(nn.Module):
    def __init__(self, channel):
        super(FM, self).__init__()
        self.rcab = RCAB(channel)
        self.rcab1 = RCAB(channel)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()

    def forward(self, x):

        P = self.rcab(x)
        P1 = self.rcab1(x)

        P = P * self.alpha
        P1 = 1 - P1 * self.beta

        P1 = self.bn1(P1)
        P1 = self.relu1(P1)

        P = P + P1
        P = self.bn(P)
        P = self.relu(P)

        #print(self.alpha, self.beta)
        return P


class Fix_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Fix_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.conv4 = SimpleASPP(64,64)
        self.conv3 = SimpleASPP(64,64)
        self.conv2 = SimpleASPP(64,64)
        self.conv1 = SimpleASPP(64,64)

        self.racb4 = RCAB(channel * 4)
        self.fm4 = FM(channel * 4)

        self.cls_layer = SimpleASPP(4 * 64,64)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x1,x2,x3,x4):

        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        conv4321 = torch.cat((conv1_feat, self.upsample2(conv2_feat),self.upsample4(conv3_feat), self.upsample8(conv4_feat)),1)
        conv4321 = self.fm4(conv4321)

        sal_pred = self.cls_layer(conv4321)


        return sal_pred


# 位置注意力和空间注意力
class CAM_Module(nn.Module):
    """ Channel attention module"""
    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class SAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SAMLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool_p = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_n = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du_p = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_du_n = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, w_p, w_n):
        x_p = x * w_p
        y_p = self.avg_pool_p(x_p)
        y_p = self.conv_du_p(y_p)

        x_n = x * w_n
        y_n = self.avg_pool_n(x_n)
        y_n = self.conv_du_n(y_n)
        return x + x_p * y_p + x_n * y_n


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features,factor=1):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.factor = factor

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output

class IAM(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(IAM, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = [1, 1, 1], [1, 2, 3], [1, 3, 7]
        modules.append(IAMConv(in_channels, in_channels, rate1, 3, [1, 1, 1]))
        modules.append(IAMConv(in_channels, in_channels, rate2, 3, [1, 2, 3]))
        modules.append(IAMConv(in_channels, in_channels, rate3, 5, [2, 6, 14]))
        modules.append(IAMPooling(in_channels, in_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        res, full = [], []
        for conv in self.convs:
            f = conv(x)
            res.append(x - f)
            full.append(f)
        full = torch.cat(full, dim=1)
        res = torch.cat(res, dim=1)
        return self.project(full) + self.project(res)


class IAMConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, padding):
        modules = [
            nn.Conv2d(in_channels, in_channels // 2, kernel_size, padding=padding[0], dilation=dilation[0], bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size, padding=padding[1], dilation=dilation[1],
                      bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size, padding=padding[2], dilation=dilation[2], bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        ]
        super(IAMConv, self).__init__(*modules)


class IAMPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(IAMPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_channels

    def forward(self, x):
        size = x.shape[-2:]
        x = super(IAMPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()

        # grouping method is the only difference here
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        x1 = self.g_conv1(q)
        x2 = self.g_conv2(q)
        x3 = self.g_conv3(q)
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)

class FeatureGudianceFusion(nn.Module):
    def __init__(self, channel, M, N):
        super(FeatureGudianceFusion, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel * 2, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel * 2, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel * 2, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        # transmit the gradient cues into the context embeddings
        q3 = self.feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def feature_grouping(self, xr, xg, M):
        if M == 1:
            q = torch.cat(
                (xr, xg), 1)
        elif M == 2:
            xr_g = torch.chunk(xr, 2, dim=1)
            xg_g = torch.chunk(xg, 2, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1]), 1)
        elif M == 4:
            xr_g = torch.chunk(xr, 4, dim=1)
            xg_g = torch.chunk(xg, 4, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3]), 1)
        elif M == 8:
            xr_g = torch.chunk(xr, 8, dim=1)
            xg_g = torch.chunk(xg, 8, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7]), 1)
        elif M == 16:
            xr_g = torch.chunk(xr, 16, dim=1)
            xg_g = torch.chunk(xg, 16, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15]), 1)
        elif M == 32:
            xr_g = torch.chunk(xr, 32, dim=1)
            xg_g = torch.chunk(xg, 32, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15],
                 xr_g[16], xg_g[16], xr_g[17], xg_g[17], xr_g[18], xg_g[18], xr_g[19], xg_g[19],
                 xr_g[20], xg_g[20], xr_g[21], xg_g[21], xr_g[22], xg_g[22], xr_g[23], xg_g[23],
                 xr_g[24], xg_g[24], xr_g[25], xg_g[25], xr_g[26], xg_g[26], xr_g[27], xg_g[27],
                 xr_g[28], xg_g[28], xr_g[29], xg_g[29], xr_g[30], xg_g[30], xr_g[31], xg_g[31]), 1)
        else:
            raise Exception("Invalid Group Number!")

        return q

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
'''a –> the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
   mode –> either 'fan_in' (default) or 'fan_out'. 
Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
Choosing 'fan_out' preserves the magnitudes in the backwards pass.
'''

class FusionBlock(nn.Module):
    def __init__(self, channel):
        super(FusionBlock, self).__init__()
        self.conv_zt_e = ConvBR(channel, channel, 3, padding=1)
        self.conv_zt_e2 = ConvBR(channel * 2, channel * 2, 3, padding=1)

        self.conv_zt_u = ConvBR(channel, channel, 3, padding=1)
        self.conv_zt_u2 = ConvBR(channel * 2, channel * 2, 3, padding=1)
        self.conv_zt_dr = DimensionalReduction(8 * channel, channel)
    def forward(self, zt_e, zt_u):
        zt_e0 = self.conv_zt_e(zt_e)
        zt_e1 = torch.cat((zt_e0, zt_u), dim=1)
        zt_e2 = self.conv_zt_e2(zt_e1)

        zt_u0 = self.conv_zt_u(zt_u)
        zt_u1 = torch.cat((zt_u0, zt_e),dim=1)
        zt_u2 = self.conv_zt_u2(zt_u1)

        out = self.conv_zt_dr(torch.concat((zt_e1, zt_e2, zt_u1, zt_u2), dim=1))

        return out

class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class UncertaintyEdgeMutualFusion(nn.Module):
    def __init__(self, channel):
        super(UncertaintyEdgeMutualFusion, self).__init__()
        self.fb_4 = FusionBlock(channel)
        self.fb_6 = FusionBlock(channel)
        self.fb_8 = FusionBlock(channel)

        self.x8_cbr = ConvBR(channel, channel, 3, padding=1)
        self.x6_cbr = ConvBR(channel, channel, 3, padding=1)
        self.x4_cbr = ConvBR(channel, channel, 3, padding=1)

        self.upsample_8_6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_6_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_ifb = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # iterative feedback block
        self.upsample_f1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_8_ifb = nn.Conv2d(channel * 2, channel, 8, 4, 2)
        self.conv_4_ifb = nn.Conv2d(channel * 2, channel, 1, 1)

        self.conv_86 = ConvBR(channel * 2, channel, 3, padding=1)
        self.iter_864 = ConvBR(channel * 2, channel, 3, padding=1)
        self.out_cbr = ConvBR(channel * 2, channel, 3, padding=1)

        self.conv_out = nn.Conv2d(channel, 1, 1)
        self.f1_dr = DimensionalReduction(128, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, zt4_e, zt6_e, zt8_e, zt4_u, zt6_u, zt8_u, f1, iternum=4):
        out = []
        # size = zt4_e.size()[2:]
        out1_8 = self.fb_8(zt8_e, zt8_u)  # outx_y: x denotes interation number, y denotes layer number
        out1_6 = self.fb_6(zt6_e, zt6_u)
        out1_4 = self.fb_4(zt4_e, zt4_u)

        x8_cbr = self.x8_cbr(out1_8)
        x8_cbr = self.upsample_8_6(x8_cbr)

        x6_cbr = torch.cat((self.x6_cbr(out1_6), x8_cbr), dim=1)
        x6_cbr = self.conv_86(x6_cbr)
        x6_cbr = self.upsample_6_4(x6_cbr)

        x4_cbr = torch.cat((self.x4_cbr(out1_4), x6_cbr), dim=1)

        iter = self.iter_864(x4_cbr)

        y_out = self.upsample_f1(iter)
        y_out = torch.cat((f1, y_out), dim=1)
        y_out = self.out_cbr(y_out)
        y_out = self.conv_out(y_out)

        out.append(self.upsample(y_out))

        for _ in range(1, iternum):
            outx_8 = self.upsample_ifb(out1_8)  # outx_y: x denotes interation number, y denotes layer number
            # outx_6 = self.fb_6(zt6_e, zt6_u)
            # outx_4 = self.fb_4(zt4_e, zt4_u)

            x8_cbr = torch.cat((outx_8, iter), dim=1)
            x8_cbr = self.x8_cbr(self.conv_8_ifb(x8_cbr))
            x8_cbr = self.upsample_8_6(x8_cbr)

            x6_cbr = torch.cat((self.x6_cbr(out1_6), x8_cbr), dim=1)
            x6_cbr = self.conv_86(x6_cbr)
            x6_cbr = self.upsample_6_4(x6_cbr)

            x4_cbr = torch.cat((out1_4, iter), dim=1)
            x4_cbr = self.conv_4_ifb(x4_cbr)
            x4_cbr = torch.cat((self.x4_cbr(x4_cbr), x6_cbr), dim=1)

            iter = self.iter_864(x4_cbr)
            y_out = self.upsample_f1(iter)
            y_out = torch.cat((f1, y_out), dim=1)
            y_out = self.out_cbr(y_out)
            y_out = self.conv_out(y_out)

            out.append(self.upsample(y_out))

        # out = torch.cat((self.upsample_8_6(out1_8), out2), dim=1)
        # out = self.conv_zt34(out)
        # out = torch.cat((self.upsample_6_4(out), out3), dim=1)
        # out = self.conv_zt345(out)

        return out[-1]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DCFM(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DCFM, self).__init__()
        self.SA = nn.ModuleList([
            SpatialAttention(),
            SpatialAttention(),
        ])
        self.CA = nn.ModuleList([
            ChannelAttention(input_dim // 2),
            ChannelAttention(input_dim // 2),
            ChannelAttention(input_dim // 2),
            ChannelAttention(input_dim // 2),
        ])
        self.conv1x1 = nn.Conv2d(input_dim * 2, output_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)

    def forward(self,feats,Gudie_positive,Gudie_negative):
        fp = feats * Gudie_positive
        fn = feats * Gudie_negative
        fp_as = self.SA[0](fp)
        fn_as = self.SA[1](fn)
        fp_as_addf = feats + fp_as
        fn_as_addf = feats + fn_as
        fp_as_addf1,fp_as_addf2 = torch.chunk(fp_as_addf,2,dim=1)
        fn_as_addf1,fn_as_addf2 = torch.chunk(fn_as_addf,2,dim=1)
        feats1,feats2 = torch.chunk(feats,2,dim=1)
        fp1_feats1 = fp_as_addf1 + feats1
        fp2_feats2 = fp_as_addf2 + feats2
        fn1_feats1 = fn_as_addf1 + feats1
        fn2_feats2 = fn_as_addf2 + feats2
        fp1_feats1_ca = self.CA[0](fp1_feats1)
        fp2_feats2_ca = self.CA[1](fp2_feats2)
        fn1_feats1_ca = self.CA[2](fn1_feats1)
        fn2_feats2_ca = self.CA[3](fn2_feats2)
        fpf = torch.cat([fp1_feats1_ca * fp2_feats2_ca,fp1_feats1_ca + fp2_feats2_ca],dim=1)
        fnf = torch.cat([fn1_feats1_ca * fn2_feats2_ca,fn1_feats1_ca + fn2_feats2_ca],dim=1)
        fnp = torch.cat([fpf + fnf,fpf * fnf],dim=1)
        ft = self.conv1x1(fnp)
        ft_add_feats = ft + feats
        f_output = self.conv3x3(ft_add_feats)
        return f_output

class AM(nn.Module):
    def __init__(self):
        super(AM, self).__init__()
        self.CA = ChannelAttention()
    def forward(self,feats1,feats2):
        feats12 = feats1 + feats2
        wei_p = self.CA(feats12)
        wei_n = 1 - wei_p
        f0_p = wei_p * feats1
        f0_n = wei_n * feats2
        f0_out = f0_p + f0_n
        return f0_out

class SAM_module(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SAM_module, self).__init__()
        self.AM = nn.ModuleList([
            AM(),
            AM(),
        ])
        self.CA = CAM_Module()
        self.conv1x1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)

    def forward(self,feats_1,feats_2):
        feats1_1,feats1_2 = torch.chunk(feats_1,2,dim=1)
        feats2_1,feats2_2 = torch.chunk(feats_2,2,dim=1)
        feats_am1 = self.AM[0](feats1_1,feats2_1)
        feats_am2 = self.AM[1](feats1_2,feats2_2)
        feats_cat = torch.cat([feats_am1,feats_am2],dim=1)
        feats_cat_conv = self.conv1x1(feats_cat)
        feats_ca = self.CA(feats_cat_conv)
        feats_fuse = feats_ca + feats_1
        feats_out = self.conv3x3(feats_fuse)
        return feats_out

# 结构注意力
class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

# 通道注意力
class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        # 低频分量 -> 全局平均池化 / 高频分量  -> 全局最大池化
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x) # tensor_1.expand_as(tensor_2) ：把tensor_1扩展成和tensor_2一样的形状，在这里就是把x扩展成y一样的形状

class RAM(nn.Module):
    def __init__(self, channels): # 这个效果和上面那个模块的效果差不多
        super(RAM, self).__init__()
        self.SA = SA(channels)
        self.CA = CA(channels)

    def forward(self, x):
        x_CA = self.CA(x)
        x_SA = self.SA(x)
        return x_CA, x_SA



class SCALE(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SCALE, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2d_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.RAM = RAM(out_channels)
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1)
        self.conv3x3_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1,padding=1)
        self.FeatureFusionBlock = FeatureFusionBlock(out_channels)

    def forward(self, *xs):

        if len(xs) == 1:
            ram_feats = self.FeatureFusionBlock(xs[0])
        if len(xs) == 2:
            # cat_feats = torch.cat((xs[0], xs[1]), dim=1)
            cat_feats = xs[0] + xs[1]
            cat_feats = self.sigmoid(self.conv2d_1(cat_feats))
            cat_images = xs[0] * cat_feats
            cat_edges =  xs[1] * cat_feats
            mul_feats = cat_images + cat_edges
            mul_conv_feats = self.conv2d_2(mul_feats)
            out_feats = mul_feats + mul_conv_feats
            ram_feats1,ram_feats2 = self.RAM(out_feats)
            ram_feats1 = self.conv3x3_1(ram_feats1)
            ram_feats2 = self.conv3x3_2(ram_feats2)
            ram_feats = torch.cat((ram_feats1, ram_feats2), dim=1)
            ram_feats = self.conv3x3_3(ram_feats)
            ram_feats = F.interpolate(ram_feats, scale_factor=2, mode="bilinear", align_corners=True)

        return ram_feats

