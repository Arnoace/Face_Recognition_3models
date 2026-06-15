"""陌生人拒识测试 — 评估模型对未注册人脸的误识率(FAR)

用法:
    # 用 20 类注册，8 类当陌生人测试
    python -m facenet.test_stranger \
        --model models/facenet_model.pth \
        --data-dir datasets/processed/cropped_train \
        --known-classes 20 \
        --threshold 0.5

    # 扫描多个阈值，找最佳平衡点
    python -m facenet.test_stranger \
        --model models/facenet_model.pth \
        --data-dir datasets/processed/cropped_train \
        --known-classes 20 \
        --scan
"""

import argparse
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .model import load_model
from .dl_utils import FaceDataset, get_test_transforms


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
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


def test_stranger(model_path: Path, data_dir: Path, known_classes: int = 20,
                  threshold: float = 0.5, scan: bool = False, seed: int = 42):
    """陌生人拒识测试

    思路: 从全部类中随机抽 known_classes 个作为"已注册人员"(gallery)，
    其余类全部作为"陌生人"，测试在给定阈值下:
      - TAR (True Acceptance Rate):  已注册人员的正确识别率
      - FAR (False Acceptance Rate):  陌生人被误识别为某人的比例
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"模型: {model_path}")

    model = load_model(model_path, device=device)
    model.eval()

    # 加载数据集
    dataset = FaceDataset(data_dir, transform=get_test_transforms(), augment=False)
    all_classes = dataset.classes
    all_labels_list = [dataset.samples[i][1] for i in range(len(dataset))]

    # 随机划分已知/未知类
    random.seed(seed)
    shuffled = list(range(len(all_classes)))
    random.shuffle(shuffled)
    known_class_ids = set(shuffled[:known_classes])
    unknown_class_ids = set(shuffled[known_classes:])

    # 提取所有 embedding
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    all_embs, _ = extract_embeddings(model, loader, device)

    # 按类别划分
    known_indices = [i for i, l in enumerate(all_labels_list) if l in known_class_ids]
    unknown_indices = [i for i, l in enumerate(all_labels_list) if l in unknown_class_ids]

    known_embs = all_embs[known_indices]          # gallery + probe
    known_labels = torch.tensor([all_labels_list[i] for i in known_indices])
    unknown_embs = all_embs[unknown_indices]       # 陌生人

    print(f"\n已注册人员: {known_classes} 类, {len(known_embs)} 张图片")
    print(f"陌生人:     {len(all_classes) - known_classes} 类, {len(unknown_embs)} 张图片")
    print(f"阈值:       {threshold}")

    if scan:
        thresholds = np.arange(0.2, 0.9, 0.025)
        _scan_thresholds(known_embs, known_labels, unknown_embs, thresholds)
    else:
        _report_threshold(known_embs, known_labels, unknown_embs, threshold)


def _report_threshold(known_embs, known_labels, unknown_embs, threshold):
    """在单个阈值下计算 TAR 和 FAR"""
    known_embs = F.normalize(known_embs, p=2, dim=1)
    unknown_embs = F.normalize(unknown_embs, p=2, dim=1)

    # --- 已知人员: TAR (留一法验证) ---
    n_known = len(known_embs)
    tar_correct = 0
    for i in range(n_known):
        # 以第 i 张为 probe，其余为 gallery
        probe = known_embs[i:i+1]
        gallery = torch.cat([known_embs[:i], known_embs[i+1:]])
        gallery_labels = torch.cat([known_labels[:i], known_labels[i+1:]])

        sim = torch.mm(probe, gallery.t()).squeeze(0)
        max_sim, max_idx = sim.max(dim=0)

        if max_sim.item() >= threshold:
            pred_label = gallery_labels[max_idx].item()
            true_label = known_labels[i].item()
            if pred_label == true_label:
                tar_correct += 1

    tar = tar_correct / n_known if n_known > 0 else 0

    # --- 陌生人: FAR ---
    n_unknown = len(unknown_embs)
    far_false = 0
    gallery = known_embs

    sim_matrix = torch.mm(unknown_embs, gallery.t())  # (U, N_known)
    max_sims, _ = sim_matrix.max(dim=1)  # 每个陌生人最高相似度

    far_false = (max_sims >= threshold).sum().item()
    far = far_false / n_unknown if n_unknown > 0 else 0

    # 输出
    print(f"\n{'='*55}")
    print(f"  阈值: {threshold:.3f}")
    print(f"{'='*55}")
    print(f"  TAR (已注册正确识别): {tar:.4f} ({tar_correct}/{n_known})")
    print(f"  FAR (陌生人误识率):   {far:.4f} ({far_false}/{n_unknown})")
    print(f"  陌生人最高相似度:     {max_sims.max().item():.4f}")
    print(f"  陌生人平均最高相似度: {max_sims.mean().item():.4f}")
    print(f"{'='*55}")
    return tar, far


def _scan_thresholds(known_embs, known_labels, unknown_embs, thresholds):
    """扫描多个阈值，输出 TAR/FAR 表格"""
    known_embs = F.normalize(known_embs, p=2, dim=1)
    unknown_embs = F.normalize(unknown_embs, p=2, dim=1)
    n_known = len(known_embs)
    n_unknown = len(unknown_embs)

    # 预计算：每个已知人的最高相似度（留一法）
    print("\n预计算已知人员相似度...")
    known_max_sims = []
    for i in tqdm(range(n_known), desc="已知人员"):
        probe = known_embs[i:i+1]
        gallery = torch.cat([known_embs[:i], known_embs[i+1:]])
        gallery_labels = torch.cat([known_labels[:i], known_labels[i+1:]])
        sim = torch.mm(probe, gallery.t()).squeeze(0)
        # 同类最高相似度（用于 TAR）
        true_label = known_labels[i]
        same_class_mask = (gallery_labels == true_label)
        if same_class_mask.any():
            known_max_sims.append(sim[same_class_mask].max().item())
        else:
            known_max_sims.append(0.0)

    # 陌生人：最高相似度
    print("预计算陌生人相似度...")
    sim_matrix = torch.mm(unknown_embs, known_embs.t())
    unknown_max_sims, _ = sim_matrix.max(dim=1)

    known_max_sims = np.array(known_max_sims)
    unknown_max_sims = unknown_max_sims.numpy()

    # 找 EER 附近的阈值
    print(f"\n{'='*80}")
    print(f"  {'阈值':>6} | {'TAR':>8} | {'FAR':>8} | {'正确/已知':>14} | {'误识/陌生人':>14} | {'陌生人最高分':>14}")
    print(f"{'='*80}")

    best_eer_idx = 0
    best_eer_diff = float('inf')

    for t in thresholds:
        tar = (known_max_sims >= t).mean()
        far = (unknown_max_sims >= t).mean()
        eer_diff = abs(tar - (1 - far))

        n_tar = (known_max_sims >= t).sum()
        n_far = (unknown_max_sims >= t).sum()

        marker = " ***" if eer_diff < 0.02 else ""
        print(f"  {t:>6.3f} | {tar:>7.2%} | {far:>7.2%} | {n_tar:>5}/{n_known:<5} | {n_far:>5}/{n_unknown:<5} | {unknown_max_sims.max():>10.4f}{marker}")

        if eer_diff < best_eer_diff:
            best_eer_diff = eer_diff
            best_eer_idx = np.where(thresholds == t)[0][0]

    best_t = thresholds[best_eer_idx]
    best_tar = (known_max_sims >= best_t).mean()
    best_far = (unknown_max_sims >= best_t).mean()

    print(f"{'='*80}")
    print(f"\n推荐阈值: {best_t:.3f} (TAR={best_tar:.2%}, FAR={best_far:.2%})")
    print(f"  如果要低 FAR (更安全): 选阈值 {best_t + 0.1:.3f} (FAR≈{(unknown_max_sims >= best_t+0.1).mean():.2%})")
    print(f"  如果要高 TAR (更方便): 选阈值 {best_t - 0.1:.3f} (TAR≈{(known_max_sims >= best_t-0.1).mean():.2%})")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="FaceNet 陌生人拒识测试")
    parser.add_argument("--model", type=Path, default=Path("models/facenet_model.pth"),
                       help="模型路径")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="数据集目录（将自动划分为已知/陌生人）")
    parser.add_argument("--known-classes", type=int, default=20,
                       help="用多少类作为已注册人员 (默认 20，其余当陌生人)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="识别阈值")
    parser.add_argument("--scan", action='store_true',
                       help="扫描多个阈值，输出 TAR/FAR 表格")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，保证可重复")
    args = parser.parse_args()

    test_stranger(args.model, args.data_dir,
                  known_classes=args.known_classes,
                  threshold=args.threshold,
                  scan=args.scan,
                  seed=args.seed)


if __name__ == "__main__":
    main()
