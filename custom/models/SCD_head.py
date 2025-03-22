import torch
import torch.nn as nn
from torch import Tensor
from mmseg.models.builder import LOSSES
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from .utils import Representations, refine, LightAttention,Fusion,Rresidual,Up_style,Up_content,ResidualBlock
from mmseg.ops import resize
from typing import Tuple
from mmseg.models.builder import HEADS
from pytorch_wavelets import DWTForward

@HEADS.register_module()
class SCDHead(BaseModule):
    def __init__(
        self,
        channels,
        in_channels=3,
        base_channels=32,
        num_downsample=2,
        num_resblock=2,
        attn_channels=128,
        image_pool_channels=128,
        ill_embeds_op="+",
        clip=True,
        gray_illumination=False,
        eps=1e-5,
        loss_dig=None,
        loss_smooth=None,
        conv_cfg=None,
        norm_cfg=dict(type="IN2d"),
        act_cfg=dict(type="ReLU"),
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01),
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.fp16_enabled = False
        self.loss_dig = LOSSES.build(loss_dig)
        self.loss_smooth = LOSSES.build(loss_smooth)
        self.eps = eps
        self.init_autoencoder(
            base_channels,
            num_downsample,
            num_resblock,
            attn_channels,
            image_pool_channels,
        )
        self.content_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )
        self.style_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )
        self.ill_embeds_op = ill_embeds_op
        self.clip = clip
        self.gray_illumination = gray_illumination

    def init_autoencoder(
        self,
        base_channels,
        num_downsample,
        num_resblock,
        attn_channels,
        image_pool_channels,
    ):
        assert (
            num_resblock >= 1
            and num_downsample >= 1
            and attn_channels >= 1
            and image_pool_channels >= 1
        )
        channels = base_channels
        self.stem = ConvModule(
            self.in_channels,
            channels,
            kernel_size=7,
            padding=3,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        down_layers = []
        for _ in range(num_downsample):
            down_layers += [
                ConvModule(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            ]
            channels *= 2
        self.downsample = nn.Sequential(*down_layers)
        res_layers = []
        for _ in range(num_resblock):
            res_layers += [
                ResidualBlock(
                    channels,
                    self.conv_cfg,
                    self.act_cfg,
                    self.norm_cfg,
                    "reflect",
                )
            ]
        self.residual=Rresidual(dim=128,kernel_size=3)
        self.light_attention = LightAttention(128, attn_channels, pre_downsample=2)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, image_pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.merge = ConvModule(
            channels + attn_channels + image_pool_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.upsample_style=Up_style(dim=128,kernel_size=3)
        self.upsample_content=Up_content(dim=32,kernel_size=3)
        self.refine = refine(dim=32,kernel_size=3)
        self.fus=Fusion(dim=128,reduction=8)

    def GLFE(self, img_embeds):
        feat0 = self.downsample(img_embeds)
        feats = [feat0]
        feats += [
            self.light_attention(feat0),
        ]
        feats = self.fus(feats[0], feats[1])
        return feats
    
    def _forward_feature(self, imgs: Tensor) -> Tensor:
        img_embeds = self.stem(imgs)
        feats = self.GLFE(img_embeds)
        feats = self.residual(feats)
        style_embeds = self.upsample_style(feats)
        content_embeds = self.upsample_content(feats)
        if self.ill_embeds_op == "+":
            content_embeds = self.refine(content_embeds + style_embeds + img_embeds)
        elif self.ill_embeds_op == "-":
            content_embeds = self.refine(content_embeds - style_embeds + img_embeds)
        return content_embeds, style_embeds, feats

    @auto_fp16(apply_to=("imgs",))
    def forward(self, imgs: Tensor) -> Representations: 
        content_embeds, style_embeds, feats = self._forward_feature(
            torch.cat([imgs, torch.max(imgs, dim=1, keepdim=True).values], dim=1)
        )
        style = self.style_output(style_embeds)
        style = torch.mean(style, dim=1, keepdim=True).repeat(1, 3, 1, 1) 
        content = self.content_output(content_embeds) + imgs 
        return Representations(style, content, feats, clip=self.clip)

    def forward_train(self, imgs: Tensor) -> Tuple[Representations, dict]:
        repres = self.forward(imgs)
        losses = dict(
            loss_smooth=self.loss_smooth(repres.style, repres.content),
            loss_dig=self.loss_dig(repres.rebuild(), imgs),
        )
        return repres, losses
