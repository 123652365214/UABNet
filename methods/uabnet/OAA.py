import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from torch import einsum
from functools import partial
from math import sqrt

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Unfold):
            pass
        elif isinstance(m, GELU):
            pass
        else:
            m.initialize()

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=False)

class dilatedComConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1):
        super(dilatedComConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[0]//2,
                            stride=stride,dilation=1)
        self.conv2_2 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[1]//2,
                            stride=stride,dilation=2)
        self.conv2_3 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[2]//2,
                            stride=stride,dilation=3)
        self.conv2_4 = conv(inplans, planes//4, kernel_size=3, padding=pyconv_kernels[3]//2,
                            stride=stride,dilation=4)

    def forward(self, x):
        conv2_1 = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        conv2_3 = self.conv2_3(x)
        conv2_4 = self.conv2_4(x)
        return torch.cat((conv2_1, conv2_2, conv2_3, conv2_4), dim=1)

    def initialize(self):
        weight_init(self)

class LSCE(nn.Module):
    def __init__(self, BatchNorm, inplanes, planes, reduction1=4):
        super(LSCE, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            dilatedComConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)

    def initialize(self):
        weight_init(self)

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.reduction_ratio = reduction_ratio

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:] # h:64,w:64
        heads, r = self.heads, self.reduction_ratio # heads:1 r:8
        # to_qkv:Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) # x:[1,32,64,64]->[1,96,64,64]->3*[1,32,64,64]
        # k, v = map(lambda t: reduce(t, 'b c (h r1) (w r2) -> b c h w', 'mean', r1 = r, r2 = r), (k, v))
        # k,v : [1,32,64,64] -> [1,32,8,8]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        # 分为multi-head 此时head数为1
        # q:[1,32,64,64] -> [1,4096,32]
        # k:[1,32,8,8] -> [1,64,32]
        # v:[1,32,8,8] -> [1,64,32]
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q*k: [1,4096,32]*[1,64,32]->[1,4096,64]
        attn = sim.softmax(dim = -1)
        # attention:[1,4096,64]
        out = einsum('b i j, b j d -> b i d', attn, v)
        # attn*v: [1,4096,64]*[1,64,32]->[1,4096,32]
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        # out: [1,4096,32]->[1,32,64,64]
        # to_out:Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # output:[1,32,64,64]->[1,32,64,64]
        return self.to_out(out)
    def initialize(self):
        weight_init(self)
class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            GELU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )

        # MixFeedForward(
        #  Sequential(
        # (0): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (2): GELU()
        # (3): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        # )
        # )

    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

class GSSE(nn.Module):
    def __init__(self, BatchNorm, inplanes, planes, reduction1=4):
        super(GSSE, self).__init__()
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True)
        )
        self.get_overlap_patches = nn.Unfold(3, dilation=1, stride=2, padding=1)

        self.overlap_embed = nn.Conv2d(inplanes // reduction1 * 9, planes, kernel_size=(1, 1), stride=(1, 1))
        self.SelfAttention = EfficientSelfAttention(dim=planes, heads=4, reduction_ratio=2)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=4)
        self.LN = LayerNorm(planes)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]
        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)
        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    def initialize(self):
        weight_init(self)

# class OAA(nn.Module):
#     def __init__(self,BatchNorm,cur_in_channels=64,low_in_channels=128,out_channels=64,cur_scale=2,low_scale=1,):
#         super(OAA,self).__init__()
#         self.cur_in_channels = cur_in_channels
#         # self.cur_conv = nn.Sequential(
#         #     nn.Conv2d(in_channels=cur_in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
#         #     nn.BatchNorm2d(num_features=out_channels),
#         #     nn.GELU()
#         # )
#         self.cur_conv = LSCE(BatchNorm,cur_in_channels,out_channels,reduction1=4)
#         # self.low_conv = nn.Sequential(
#         #     nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm2d(num_features=out_channels),
#         #     nn.GELU()
#         # )
#         self.low_conv = GSSE(BatchNorm,low_in_channels,out_channels,reduction1=4)
#
#         self.cur_scale = cur_scale
#         self.low_scale = low_scale
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(num_features=out_channels),
#             # nn.GELU()
#         )
#
#     def forward(self,x_cur,x_low):
#         x_cur = self.cur_conv(x_cur)
#         #bicubic bilinear nearest
#         x_cur = F.interpolate(x_cur, scale_factor=self.cur_scale,  mode='bicubic',align_corners = False)
#
#         x_low = self.low_conv(x_low)
#         x_low = F.interpolate(x_low, scale_factor=self.low_scale,  mode='bicubic',align_corners = False)
#         x = torch.cat((x_cur,x_low),dim=1)
#         x = self.out_conv(x)
#         return x
from .ops import ConvBNReLU
class VFMRM(nn.Module):
    def __init__(self,BatchNorm,in_channels,out_channels):
        super().__init__()
        self.GSSE = GSSE(BatchNorm,in_channels,out_channels,reduction1=4)
        self.LSCE = LSCE(BatchNorm,in_channels,out_channels,reduction1=4)
        self.channl = ConvBNReLU(in_channels*2 ,in_channels,3,1,1)
    def forward(self, x):
        x1 = self.GSSE(x)
        x2 = self.LSCE(x)
        input = torch.cat((x1,x2),1)
        output = self.channl(input)
        return output




if __name__ == '__main__':
    x1 = torch.randn(4, 128, 96, 96)
    x2 = torch.randn(4, 128, 48, 48)
    x3 = torch.randn(4, 320, 24, 24)
    x4 = torch.randn(4, 512, 12, 12)
    oaa = OAA(nn.BatchNorm2d,320, 512,64, cur_scale=1, low_scale=2)
    output = oaa(x3,x4)
    print(f"输出形状为：{output.shape}")
