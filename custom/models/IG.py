import torch
import torch.nn as nn
from mmseg.models.decode_heads import UPerHead
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from mmseg.core import add_prefix
from mmseg.models.decode_heads.psp_head import PPM
from .utils import StripPooling

@HEADS.register_module()
class IG(UPerHead):
    def __init__(self, style_channels, style_features_channels, **kwargs):
        super().__init__(**kwargs)
        self.style_transform = nn.Sequential(
            ConvModule(
                style_channels,
                style_features_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            ConvModule(
                style_features_channels,
                style_features_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        self.style_ppm = PPM(
            (1, 2, 3, 6),
            style_features_channels,
            style_features_channels // 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )
        self.style_bottleneck = ConvModule(
            style_features_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.conv_style_seg = nn.Conv2d(self.channels, self.num_classes, 1, 1)
        self.mask_style_layer = nn.Sequential(
            ConvModule(
                style_features_channels + self.channels,
                style_features_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            nn.Conv2d(
                style_features_channels,
                self.channels,
                kernel_size=1,
                stride=1,
            ),
            nn.Sigmoid(),
        )
        self.conv_all_seg = nn.Conv2d(self.channels, self.num_classes, 1, 1)
        self.strippool = StripPooling(in_channels=self.channels)

    def forward(self, content_feats, style_feats, use_for_loss=False):
        seg_feats_content = self._forward_feature(content_feats)
        logits_content = self.cls_seg(seg_feats_content)
        with torch.no_grad():
            style_feats = resize(
                style_feats,
                seg_feats_content.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        style_feats = self.style_transform(style_feats)
        seg_feats_style = self.style_bottleneck(
            torch.cat([style_feats] + self.style_ppm(style_feats), dim=1)
        )
        logits_style = self.conv_style_seg(seg_feats_style)
        mask_style = self.mask_style_layer(torch.cat([style_feats, seg_feats_content], dim=1))
        mask_style = self.strippool(mask_style)
        seg_feats_whole = seg_feats_content + seg_feats_style * mask_style
        logits_whole = self.conv_all_seg(seg_feats_whole)
        if not use_for_loss:
            return logits_whole
        else:
            return logits_whole, logits_content, logits_style

    def losses(self, seg_logits, seg_label):
        logits_whole, logits_content, logits_style = seg_logits
        losses = dict()
        losses.update(add_prefix(super().losses(logits_content, seg_label), "logits_content"))
        losses.update(
            add_prefix(super().losses(logits_whole, seg_label), "logits_whole")
        )
        losses.update(add_prefix(super().losses(logits_style, seg_label), "logits_style"))
        return losses