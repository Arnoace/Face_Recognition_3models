"""Google FaceNet 模型 — Inception (GoogLeNet) + Triplet Embedding
符合原版 FaceNet 论文 (Schroff et al., CVPR 2015):
  - 骨干网络: Inception (GoogLeNet)，即原版 NN2 架构
  - 嵌入维度: 512 维 (原版 128，社区标准 512)
  - 归一化: L2 归一化
  - 损失函数: Triplet Loss (见 dl_losses.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
import numpy as np

class FaceNetModel(nn.Module):
    """FaceNet 模型 — GoogLeNet (Inception v1) 骨干 + BNNeck Embedding
    原版 FaceNet NN2 采用 GoogLeNet 风格的 Inception 架构，
    这里使用 torchvision 的 GoogLeNet 作为特征提取器。
    """
    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()

        # GoogLeNet (Inception v1) 骨干网络
        # 原版 FaceNet NN2 正是基于 GoogLeNet 架构
        googlenet = models.googlenet(
            weights='GoogLeNet_Weights.IMAGENET1K_V1' if pretrained else None
        )

        # GoogLeNet 输入要求 RGB 3 通道 (3, 224, 224)
        # 取出除最后分类层之外的所有层
        # googlenet 包含: conv1, maxpool1, conv2, conv3, maxpool2,
        # inception3a~3b, maxpool3, inception4a~4e, maxpool4,
        # inception5a~5b, avgpool, dropout, fc
        # 我们取到 avgpool (去掉 dropout 和 fc)
        self.backbone = nn.Sequential(
            googlenet.conv1,
            googlenet.maxpool1,
            googlenet.conv2,
            googlenet.conv3,
            googlenet.maxpool2,
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3,
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a,
            googlenet.inception5b,
            googlenet.avgpool,
        )

        # GoogLeNet 最终输出 1024 维特征
        self.embedding_size = embedding_size
        self.bn = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 其中 C=1(灰度) 或 C=3(RGB)
        Returns:
            embedding: (B, embedding_size) L2 归一化特征向量
        """
        # 灰度图 (1 通道) → 复制成 3 通道 RGB
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # GoogLeNet 需要 224x224 输入，做插值
        if x.size(-1) != 224 or x.size(-2) != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # 特征提取
        features = self.backbone(x)          # (B, 1024, 1, 1)
        features = torch.flatten(features, 1)  # (B, 1024)

        # 嵌入层
        embedding = self.bn(features)
        embedding = self.fc(embedding)          # (B, embedding_size)

        # L2 归一化（原版 FaceNet 的核心）
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def extract_feature(self, x):
        """对外接口：提取归一化特征向量（用于比对）"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.dim() == 3:
                x = x.unsqueeze(0)   # (H, W, C) → (1, H, W, C)
            # 确保通道维度正确
            if x.dim() == 4:
                if x.size(1) != 1 and x.size(1) != 3:
                    x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
            else:
                x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            return self.forward(x)


# ==================== 保存/加载工具 ====================
def save_model(model: FaceNetModel, path: Path, classes: list = None):
    """保存 FaceNet 模型"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'bn_state_dict': model.bn.state_dict(),
        'fc_state_dict': model.fc.state_dict(),
        'embedding_size': model.embedding_size,
        'classes': classes or [],
    }, path)
    print(f"[完成] FaceNet 模型已保存: {path}")


def load_model(path: Path, device='cpu'):
    """加载 FaceNet 模型"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = FaceNetModel(
        embedding_size=checkpoint.get('embedding_size', 512),
        pretrained=False
    )

    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.bn.load_state_dict(checkpoint['bn_state_dict'])
    model.fc.load_state_dict(checkpoint['fc_state_dict'])
    model.classes = checkpoint.get('classes', [])
    model.to(device)
    model.eval()
    print(f"[完成] FaceNet 模型加载成功，类别数: {len(model.classes)}，嵌入维度: {model.embedding_size}")
    return model
