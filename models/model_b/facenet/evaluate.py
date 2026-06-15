"""FaceNet 模型评估 — 用余弦相似度 + K 近邻分类

Inception-ResNet-v1 输出 L2 归一化的 embedding，
用余弦相似度 + KNN 做分类。
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model import load_model
from .dl_utils import FaceDataset, get_test_transforms


@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    """提取数据集中所有图片的 embedding 和对应标签"""
    model.eval()
    all_embs = []
    all_labels = []
    for imgs, labels in tqdm(dataloader, desc="提取特征"):
        imgs = imgs.to(device)
        embs = model.extract_feature(imgs)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


def evaluate_knn(gallery_embs, gallery_labels, probe_embs, probe_labels, k=5):
    """用 K 近邻（余弦相似度）做分类并计算 top-1 / top-5 准确率

    Args:
        gallery_embs: (N_gallery, D) tensor, 参考集特征
        gallery_labels: (N_gallery,) tensor, 参考集标签
        probe_embs: (N_probe, D) tensor, 测试集特征
        probe_labels: (N_probe,) tensor, 测试集标签
        k: 最近邻数
    Returns:
        top1_acc, top5_acc
    """
    # L2 归一化（确保余弦相似度 = dot product）
    gallery_embs = F.normalize(gallery_embs, p=2, dim=1)
    probe_embs = F.normalize(probe_embs, p=2, dim=1)

    # 相似度矩阵 (N_probe, N_gallery)
    sim = torch.mm(probe_embs, gallery_embs.t())

    # 取 top-k 最近邻
    topk_sim, topk_idx = sim.topk(k=k, dim=1)  # (N_probe, k)

    topk_labels = gallery_labels[topk_idx]  # (N_probe, k)

    # Top-1: 最近邻的标签
    pred_top1 = topk_labels[:, 0]
    top1_correct = (pred_top1 == probe_labels).sum().item()

    # Top-5: k 个近邻中任意一个匹配
    top5_correct = (topk_labels == probe_labels.unsqueeze(1)).any(dim=1).sum().item()

    total = probe_labels.size(0)
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    return top1_acc, top5_acc


def evaluate(model_path: Path, train_dir: Path, test_dir: Path, batch_size: int = 64, k: int = 5):
    """完整评估流程：从训练集构建 gallery，在测试集上做 KNN 分类"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = load_model(model_path, device=device)

    # 加载训练集（做 gallery / 参考集）
    try:
        train_dataset = FaceDataset(train_dir, transform=get_test_transforms(), augment=False)
    except Exception as e:
        print(f"[错误] 加载训练集失败: {e}")
        return

    if len(train_dataset) == 0:
        print("[警告] 训练集为空!")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 加载测试集（做 probe / 探针）
    try:
        test_dataset = FaceDataset(test_dir, transform=get_test_transforms(), augment=False)
    except Exception as e:
        print(f"[错误] 加载测试集失败: {e}")
        return

    if len(test_dataset) == 0:
        print("[警告] 测试集为空!")
        return

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\n训练集 (gallery): {len(train_dataset)} 张图片")
    print(f"测试集 (probe):   {len(test_dataset)} 张图片")
    print(f"类别数: {len(train_dataset.classes)}")

    # 提取所有 embedding
    print("\n[1/2] 提取训练集 gallery 特征...")
    gallery_embs, gallery_labels = extract_all_embeddings(model, train_loader, device)
    print(f"  gallery 特征矩阵: {gallery_embs.shape}")

    print("[2/2] 提取测试集 probe 特征...")
    probe_embs, probe_labels = extract_all_embeddings(model, test_loader, device)
    print(f"  probe 特征矩阵: {probe_embs.shape}")

    # KNN 评估
    print(f"\n使用 k={k} 最近邻（余弦相似度）评估...")
    top1_acc, top5_acc = evaluate_knn(
        gallery_embs, gallery_labels,
        probe_embs, probe_labels,
        k=k
    )

    print("\n" + "=" * 50)
    print(f"  Top-1 准确率: {top1_acc:.4f} ({int(top1_acc * probe_labels.size(0))}/{probe_labels.size(0)})")
    print(f"  Top-5 准确率: {top5_acc:.4f} ({int(top5_acc * probe_labels.size(0))}/{probe_labels.size(0)})")
    print("=" * 50)

    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="FaceNet 模型评估（余弦相似度 KNN）")
    parser.add_argument("--model", type=Path, required=True, help="模型路径")
    parser.add_argument("--train-dir", type=Path, required=True, help="训练集目录（构建 gallery）")
    parser.add_argument("--test-dir", type=Path, required=True, help="测试集目录（做 probe）")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--k", type=int, default=5, help="KNN 的 k 值")
    args = parser.parse_args()

    evaluate(args.model, args.train_dir, args.test_dir,
             batch_size=args.batch_size, k=args.k)


if __name__ == "__main__":
    main()