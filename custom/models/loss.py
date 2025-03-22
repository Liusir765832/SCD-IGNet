import torch
import torch.nn as nn
from torchvision import transforms
from mmseg.models import LOSSES


@LOSSES.register_module()
class ContentLoss(nn.Module):
    def __init__(self, weight=1):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.SmoothL1Loss()

    def forward(self, predicate,target):
        self.target = target.detach() * self.weight
        losses = dict(
            content_loss= self.criterion(predicate * self.weight, self.target),
        )
        return losses

class GramMatrix(nn.Module):
    def forward(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)


@LOSSES.register_module()
class StyleLoss(nn.Module):
    def __init__(self, weight):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.gram = GramMatrix()

    def forward(self, predicate,target):
        losses = dict(
            style_loss= torch.square(self.gram(predicate) - self.gram(target).detach()).mean(),
        )
        return losses

# 主函数
def main():
    # 创建随机数据和目标
    content = torch.randn(4, 3, 256, 256, requires_grad=True)
    style = torch.randn(4, 3, 256, 256, requires_grad=True)
    target_content = torch.randn(4, 3, 256, 256)
    target_style = torch.randn(4, 3, 256, 256)
    gram = GramMatrix()

    # 创建权重

    # 实例化损失函数
    content_loss = ContentLoss(weight=1)
    style_loss = StyleLoss(weight=10)

    # 计算损失
    content_output = content_loss(content,target_content)
    style_output = style_loss(style,target_style)


    # 打印结果
    print(f"Content Loss:",content_output)
    print(f"Style Loss:",style_output)

# 调用主函数
if __name__ == "__main__":
    main()
