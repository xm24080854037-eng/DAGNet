import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """自动计算 padding，使得输出尺寸保持不变（即 same padding）"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """
    标准卷积模块（卷积 + BatchNorm + 激活函数）
    支持指定 kernel size、stride、padding、dilation、groups、是否使用激活函数等。
    默认激活函数为 SiLU（Swish），可设为 False 禁用激活。
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小（默认为 1）
        s: 步长
        p: padding（默认自动计算）
        g: 分组卷积参数
        d: dilation rate
        act: 是否使用激活函数（默认为 True -> 使用 SiLU）
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """用于模型部署后的融合版本（BN合并到Conv中）"""
        return self.act(self.conv(x))
