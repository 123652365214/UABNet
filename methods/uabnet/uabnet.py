import abc
import logging

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import shape
import cv2


from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .layers import OAA1,CBR,Fix_feat_decoder,SimpleASPP,PAM_Module,CAM_Module,SAMLayer,FeatureFusionBlock,IAM,FeatureGudianceFusion,ConvBR,UncertaintyEdgeMutualFusion,SCALE
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from methods.sarnet.HolisticAttention import HA
# from methods.sarnet.OAA import OAA

LOGGER = logging.getLogger("main")



def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class _sarnet_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef


    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)



        if self.training:
            mask = data["mask"]
            prob = logits.sigmoid()

            losses = []
            loss_str = []

            # sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            bce = structure_loss(prob, mask)
            dice = dice_loss(prob, mask)
            # sod_loss = bce + dice


            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            # losses.append(ual_loss)
            sod_loss = bce + dice + ual_loss
            loss_str.append(f"bce: {sod_loss.item():.5f}")
            # losses.append(sod_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
            return dict(vis=dict(sal=prob), loss=sod_loss, loss_str=" ".join(loss_str))
        else:
            return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class PvtV2B2_sarnet(_sarnet_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.embed_dims = self.encoder.embed_dims
        self.ASPP = SimpleASPP(self.embed_dims[3],self.embed_dims[0])
        self.IAM = nn.ModuleList([
            IAM(self.embed_dims[3], self.embed_dims[3]),
            IAM(self.embed_dims[2], self.embed_dims[2]),
            IAM(self.embed_dims[1], self.embed_dims[1]),
            IAM(self.embed_dims[0], self.embed_dims[0]),
            IAM(self.embed_dims[3], self.embed_dims[0]),
        ])

        self.OAA_0 = nn.ModuleList([
            OAA1(cur_in_channels=self.embed_dims[0], low_in_channels=self.embed_dims[1],
                out_channels=self.embed_dims[0], cur_scale=1, low_scale=2),
            OAA1(cur_in_channels=self.embed_dims[1], low_in_channels=self.embed_dims[2],
                out_channels=self.embed_dims[1] // 2, cur_scale=1, low_scale=2),
            OAA1(cur_in_channels=self.embed_dims[2], low_in_channels=self.embed_dims[3],
                out_channels=self.embed_dims[3] // 8, cur_scale=1, low_scale=2)
        ])

        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )


        self.CBR = CBR(in_channels=self.embed_dims[3], out_channels=self.embed_dims[3] // 8,
                       kernel_size=3, stride=1,
                       dilation=1, padding=1)

        self.Fix_feat_decoder = Fix_feat_decoder(mid_dim)
        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.OAA_1 = OAA1(cur_in_channels=self.embed_dims[0], low_in_channels=self.embed_dims[3] // 8,
                        out_channels=self.embed_dims[0], cur_scale=2,
                        low_scale=16)  #
        self.HA = HA()
        self.CAM = CAM_Module()
        self.PAM = PAM_Module(mid_dim)
        self.conv_mask = nn.Conv2d(mid_dim, 1, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.SAM = nn.ModuleList([
            SAMLayer(mid_dim),
            SAMLayer(mid_dim),
            SAMLayer(mid_dim),
            SAMLayer(mid_dim),
        ])
        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.Decoder = nn.ModuleList([
            SCALE(mid_dim,mid_dim),
            SCALE(mid_dim,mid_dim),
            SCALE(mid_dim,mid_dim),
            SCALE(mid_dim,mid_dim),
            # FeatureFusionBlock(mid_dim),
        ])
        self.CAM = nn.ModuleList([
            CAM_Module(),
            CAM_Module(),
            CAM_Module(),
            CAM_Module(),
        ])
        self.PAM = nn.ModuleList([
            PAM_Module(mid_dim),
            PAM_Module(mid_dim),
            PAM_Module(mid_dim),
            PAM_Module(mid_dim),
        ])
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.input_proj = nn.Sequential(ConvBR(mid_dim, mid_dim, kernel_size=1), nn.Dropout2d(p=0.1))
        self.conv = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.mean_conv = nn.Conv2d(mid_dim, 1, kernel_size=1, bias=False)
        self.std_conv = nn.Conv2d(mid_dim, 1, kernel_size=1, bias=False)
        self.fgf_uncertainty = FeatureGudianceFusion(channel=mid_dim, M=[8, 8, 8], N=[4, 8, 16])
        self.fgf_uncertainty = FeatureGudianceFusion(channel=mid_dim, M=[8, 8, 8], N=[4, 8, 16])
        self.uemf = UncertaintyEdgeMutualFusion(channel=64)

        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)


        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type: Variable # 1, 1, 33, 33
            # eps1 = std.data.new(std.size()).normal_() # 1, 1, 33, 33  fill with gaussion N(0, 1); change every loop
            eps = np.float32(np.random.laplace(0, 1, std.size()))
            eps = torch.from_numpy(eps).cuda()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def body(self, data):
        try:
            if data["mask"] is not None and torch.any(data["mask"] > 0):
                flag = True
            else:
                flag = False
        except KeyError:
            flag = False
        data_shape = data["image_m"] # 这个是最原始的后面的都是在这个上面改，这一份用作备份
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s1 = self.tra_5(m_trans_feats[3])
        s2 = self.tra_4(m_trans_feats[2])
        s3 = self.tra_3(m_trans_feats[1])
        s4 = self.tra_2(m_trans_feats[0])

        x_u = self.input_proj(s4)  # 1, 64, 12, 12
        mean = self.mean_conv(x_u)  # 1, 1, 12, 12
        std = self.std_conv(x_u)  # 1, 1, 12, 12

        # prob_x = self.reparameterize(mean, std, 1)
        prob_out2 = self.reparameterize(mean, std, 50)  # 1, 50, 33, 33 sample process
        prob_out2 = torch.sigmoid(prob_out2)  # 1, 50, 33, 33

        uncertainty = prob_out2.var(dim=1, keepdim=True).detach()  # 1, 1, 33, 33
        if flag:
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        uncertainty = F.interpolate(uncertainty, 96, mode='bilinear', align_corners=False)
        uncertainties = uncertainty.repeat(1, 64, 1, 1)
        zt2_u, zt3_u, zt4_u = self.fgf_uncertainty(s4, s3, s2, uncertainties) # 不确定性推理模块

        gudie_decoder = self.Fix_feat_decoder(s4,s3,s2,s1) # 粗特征提取器
        Guidance = self.conv_mask(gudie_decoder)
        Guidance_P = self.sigmoid(Guidance)
        Guidance_N = self.sigmoid(Guidance) * (-1) + 1

        SAM_1 = self.SAM[3](s1, self.down8(Guidance_P), self.down8(Guidance_N))
        SAM_4 = self.SAM[0](zt4_u,self.down4(Guidance_P),self.down4(Guidance_N))
        SAM_3 = self.SAM[1](zt3_u,self.down2(Guidance_P),self.down2(Guidance_N))
        SAM_2 = self.SAM[2](zt2_u,Guidance_P,Guidance_N)


        SAM_1 = self.CAM[0](self.PAM[0](SAM_1))
        Fusion_out4 = self.Decoder[0](SAM_1)
        Fusion_out3 = self.Decoder[1](Fusion_out4, SAM_4)
        Fusion_out2 = self.Decoder[2](Fusion_out3, SAM_3)
        Fusion_out1 = self.Decoder[3](Fusion_out2, SAM_2) # 解码器
        output = self.out_conv(Fusion_out1)
        return output


class PvtV2B3_sarnet(PvtV2B2_sarnet):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_sarnet(PvtV2B2_sarnet):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_sarnet(PvtV2B2_sarnet):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_sarnet(PvtV2B5_sarnet):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


