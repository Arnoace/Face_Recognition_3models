"""
ArcFace 骨干网络: Improved ResNet (IResNet)

这是 ArcFace 论文中使用的核心网络结构。
与标准 ResNet 的区别：
  - 使用 BN-Conv-BN-PReLU-Conv-BN 的改进残差块
  - 使用 PReLU 替代 ReLU
  - 最后一层使用 BN-Flatten-FC-BN 输出 512 维嵌入特征

论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']

using_ckpt = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 卷积 (步长可配)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积 (用于降采样 shortcut)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """
    ArcFace 改进的残差基本块 (Improved Basic Block)

    结构: BN1 → Conv3x3 → BN2 → PReLU → Conv3x3 → BN3  +  shortcut
    与标准 ResNet BasicBlock 不同：
      - BN 在 Conv 之前 (预激活/pre-activation)
      - 使用 PReLU 而非 ReLU
      - 不使用最后一层 ReLU (恒等映射)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)           # ArcFace 关键：PReLU 替代 ReLU
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)           # 预激活：先 BN
        out = self.conv1(out)       # 再 Conv
        out = self.bn2(out)         # BN
        out = self.prelu(out)       # PReLU
        out = self.conv2(out)       # Conv
        out = self.bn3(out)         # BN (无激活，恒等映射)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity              # 残差连接
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)  # 梯度检查点节省显存
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    """
    ArcFace 改进残差网络 (Improved ResNet)

    输入:   (B, 3, 112, 112)
    输出:   (B, 512) 归一化的人脸嵌入特征向量

    网络结构 (以 IResNet-50 为例):
      Conv3x3(3→64) → BN → PReLU
      → Layer1: 3× IBasicBlock(64→64, stride=2)    → (B, 64, 56, 56)
      → Layer2: 4× IBasicBlock(64→128, stride=2)   → (B, 128, 28, 28)
      → Layer3: 14× IBasicBlock(128→256, stride=2)  → (B, 256, 14, 14)
      → Layer4: 3× IBasicBlock(256→512, stride=2)   → (B, 512, 7, 7)
      → BN → Flatten → Dropout → FC(512×7×7 → 512) → BN (特征层)
    """
    fc_scale = 7 * 7  # 全连接前的空间维度: 112/(2^4) = 7

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        # ── 输入层: 3 → 64, 保持 112×112 ──
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # ── 四个残差阶段 ──
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)    # → 56×56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # → 28×28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])  # → 14×14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])  # → 7×7

        # ── 输出层 ──
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

        # 固定特征 BN 的权重为 1 (不参与梯度更新)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # ── 权重初始化 ──
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 112, 112) RGB 人脸图像 (已归一化)
        Returns:
            (B, 512) L2归一化的人脸嵌入特征向量
        """
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)       # (B, 64, 112, 112)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)      # (B, 64, 56, 56)
            x = self.layer2(x)      # (B, 128, 28, 28)
            x = self.layer3(x)      # (B, 256, 14, 14)
            x = self.layer4(x)      # (B, 512, 7, 7)
            x = self.bn2(x)
            x = torch.flatten(x, 1) # (B, 512*7*7=25088)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)   # (B, 512)
        x = self.features(x)                          # BN → 输出嵌入特征
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("Pretrained models not available in this package.")
    return model


# ── 各规格 IResNet 的层数定义 ──
# IResNet-18:   [2, 2, 2, 2]   - 轻量
# IResNet-34:   [3, 4, 6, 3]   - 中等
# IResNet-50:   [3, 4, 14, 3]  - 推荐 (精度/速度平衡)
# IResNet-100:  [3, 13, 30, 3] - 高精度
# IResNet-200:  [6, 26, 60, 6] - 最高精度 (大显存)

def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs)

def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)

def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs)
