
import torch.nn as nn
import torch
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
        
class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        # modules = []
        # modules.append(nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),#groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)
    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att+x) 



aspp = MFEblock(64,[2,4,8])
x = torch.rand(2,64,192,192)
print(aspp(x).shape)

