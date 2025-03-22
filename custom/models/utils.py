import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule
from mmseg.models.backbones.swin import SwinBlockSequence
from typing import List
import torch
from einops.layers.torch import Rearrange
from .deconv import DEConv
from torch.nn import functional as F
from torch.autograd import Variable


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size = (2, 4), norm_layer=nn.BatchNorm2d, up_kwargs = {'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):#torch.Size([2, 32, 64, 128])
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)#torch.Size([2, 8, 64, 128])
        x2 = self.conv1_2(x)#torch.Size([2, 8, 64, 128])
        x2_1 = self.conv2_0(x1)#torch.Size([2, 8, 64, 128])
        
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)#torch.Size([2, 8, 64, 128])
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)#torch.Size([2, 8, 64, 128])
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)#torch.Size([2, 8, 64, 128])
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)#torch.Size([2, 8, 64, 128])
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))#torch.Size([2, 8, 64, 128])
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))#torch.Size([2, 8, 64, 128])
        out = self.conv3(torch.cat([x1, x2], dim=1))#torch.Size([2, 32, 64, 128])
        return F.relu_(x + out)
from torch import Tensor

class Representations:
    def __init__(
        self, style: Tensor, content: Tensor, features: Tensor, clip=True
    ) -> None:
        if clip:
            self.style: Tensor = style.clip(0, 1)
            self.content: Tensor = content.clip(0, 1)
        else:
            self.style: Tensor = style
            self.content: Tensor = content
        self.features = features

    def rebuild(self, content=None, style=None) -> Tensor:
        content = self.content if content is None else content
        style = self.style if style is None else style
        return content * style


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        conv_cfg,
        act_cfg,
        norm_cfg,
        padding_mode,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode,
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                act_cfg=None,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x):
        return x + self.block(x)



class LightAttention(SwinBlockSequence):
    def __init__(
        self,
        in_channels,
        out_channels,
        pre_downsample=1,
        num_heads=4,
        window_size=8,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        init_cfg=None,
    ):
        embed_dims = out_channels * 2 * (2**pre_downsample)
        feedforward_channels = embed_dims * 4
        depth = 2
        super().__init__(
            embed_dims,
            num_heads,
            feedforward_channels,
            depth,
            window_size,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            None,
            act_cfg,
            norm_cfg,
            with_cp,
            init_cfg,
        )
        self.in_transform = ConvModule(
            in_channels, embed_dims, kernel_size=1, act_cfg=act_cfg
        )
        self.out_transform = nn.Sequential(
            nn.Linear(embed_dims, out_channels), nn.Sigmoid()
        )
        self.pre_downsample = pre_downsample
        if pre_downsample > 0:
            self.down = nn.MaxPool2d(kernel_size=pre_downsample, stride=pre_downsample)
            self.up = nn.UpsamplingBilinear2d(scale_factor=pre_downsample)

    def forward(self, x: Tensor) -> Tensor:#torch.Size([2, 128, 64, 128])
        if self.pre_downsample > 0:#下采样
            x = self.down(x)#torch.Size([2, 128, 32, 64])
        x = self.in_transform(x)#torch.Size([2, 128, 32, 64])
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(N, H * W, C)#torch.Size([2, 2048, 128])
        x, _, _, _ = super().forward(x, [H, W])
        x = self.out_transform(x)
        C = x.shape[-1]
        x = x.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        if self.pre_downsample > 0:#上采样
            x = self.up(x)
        return x


def list_detach(feats: List[Tensor]) -> List[Tensor]:
    if isinstance(feats, list):
        return [x.detach() for x in feats]
    else:
        return feats.detach()


def losses_weight_rectify(losses: dict, prefix: str, weight: float):
    for loss_name, loss_value in losses.items():
        if loss_name.startswith(prefix):
            if "loss" in loss_name:
                losses[loss_name] = loss_value * weight



"""fusion module"""
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):#torch.Size([1, 32, 224, 224])
        x_gap = self.gap(x)#torch.Size([1, 32, 1, 1])
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class Fusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(Fusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):#torch.Size([1, 32, 224, 224]),torch.Size([1, 32, 224, 224])
        initial = x + y#torch.Size([1, 32, 224, 224])
        cattn = self.ca(initial)#torch.Size([1, 32, 1, 1])
        sattn = self.sa(initial)#torch.Size([1, 1, 224, 224])
        pattn1 = sattn + cattn#torch.Size([1, 32, 224, 224])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))#torch.Size([1, 32, 224, 224])
        result = initial + pattn2 * x + (1 - pattn2) * y#torch.Size([1, 32, 224, 224])
        result = self.conv(result)
        return result

"""DEABlockTrain module and DECBlockTrain"""
class DEABlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlockTrain, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DECBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DECBlockTrain, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Rresidual(nn.Module):
    def __init__(self,dim, kernel_size):
        super(Rresidual, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size
        self.deb1=DECBlockTrain(default_conv,self.dim,self.kernel_size)
        self.deb2=DECBlockTrain(default_conv,self.dim,self.kernel_size)

    def forward(self, x):
        x=self.deb1(x)
        x=self.deb2(x)
        return x
    
class refine(nn.Module):
    def __init__(self,dim, kernel_size):
        super(refine, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size
        self.deab=DEABlockTrain(default_conv,self.dim,self.kernel_size)

    def forward(self, x):
        x=self.deab(x)
        return x

class Up_style(nn.Module):
    def __init__(self,dim, kernel_size):
        super(Up_style, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size
        self.deb1=DECBlockTrain(default_conv,self.dim,self.kernel_size)
        self.deb2=DECBlockTrain(default_conv,self.dim,self.kernel_size)
        self.upill =nn.Sequential(nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.ReLU(True),
                                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ))

    def forward(self, x):
        x=self.deb1(x)
        x=self.deb2(x)
        x=self.upill(x)
        return x

class Up_content(nn.Module):
    def __init__(self,dim, kernel_size):
        super(Up_content, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size
        self.deb1=DECBlockTrain(default_conv,self.dim,self.kernel_size)
        self.deb2=DECBlockTrain(default_conv,self.dim,self.kernel_size)
        self.upill =nn.Sequential(nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.ReLU(True),
                                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ))

    def forward(self, x): 
        x=self.upill(x)
        x=self.deb1(x)
        x=self.deb2(x)
        return x
