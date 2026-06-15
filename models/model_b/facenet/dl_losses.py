"""FaceNet 风格损失函数：CosFace + MagFace + Triplet Loss 实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class TripletLoss(nn.Module):
    """FaceNet 原版 Triplet Loss — 带 Batch Hard 采样
    原版 FaceNet (Schroff et al. 2015) 使用 Triplet Loss，
    这里实现 Batch Hard 版本 (Hermans et al. 2017)：
      对每个 anchor，取 batch 内最远的正例 + 最近的负例
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: (B, D) 已经 L2 归一化的特征向量
            labels: (B,) 类别标签，要求同一个 batch 内每类有 K>=2 张图
        Returns:
            loss: 标量
        """
        batch_size = embeddings.size(0)
        # 距离矩阵 (B, B)
        dist = 2 - 2 * torch.mm(embeddings, embeddings.t())  # 余弦距离 [0, 4]
        # 正例掩码：同类但不是自己
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
        pos_mask = label_eq.float()
        # 去掉对角线（自己和自己）
        pos_mask.fill_diagonal_(0)
        # 负例掩码：不同类
        neg_mask = (~label_eq).float()
        neg_mask.fill_diagonal_(0)
        # Batch Hard:
        #   对每个 anchor: 最远正例距离 + 最近负例距离
        hardest_pos = (dist * pos_mask).max(dim=1)[0]  # (B,) — 每个 anchor 最远的同类
        hardest_neg = (dist * neg_mask + pos_mask * 1000).min(dim=1)[0]  # (B,) — 每个 anchor 最近的异类
        # Triplet Loss: max(0, d_pos - d_neg + margin)
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        # 只统计 loss > 0 的 triplet（已经满足的不再优化）
        valid = (loss > 0).sum()
        loss = loss.sum() / max(valid, 1)
        return loss

class CosFaceLoss(nn.Module):
    """CosFace / ArcFace 风格的 Additive Margin Softmax"""
    def __init__(self, num_classes: int, embedding_size: int = 512, scale=64.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        cosine = torch.mm(embeddings, weight.t())           # [batch, num_classes]
        cosine_margin = cosine - self.margin
        
        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        logits = (one_hot * cosine_margin) + ((1 - one_hot) * cosine)
        logits *= self.scale
        
        return F.cross_entropy(logits, labels)

class MagFaceLoss(nn.Module):
    """MagFace 损失（近似实现）"""
    def __init__(self, num_classes: int, embedding_size: int = 512, scale=64.0, margin=0.5, l_a=10, u_a=110):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.l_a = l_a
        self.u_a = u_a
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        cosine = torch.mm(embeddings, weight.t())
        
        # 自适应 margin
        margin = self.margin * (embeddings_norm - self.l_a) / (self.u_a - self.l_a)
        margin = torch.clamp(margin, min=0, max=self.margin)
        
        cosine_margin = cosine - margin.squeeze(1).unsqueeze(1)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        logits = (one_hot * cosine_margin) + ((1 - one_hot) * cosine)
        logits *= self.scale
        
        return F.cross_entropy(logits, labels)