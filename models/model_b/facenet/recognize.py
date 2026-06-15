"""人脸识别脚本 — 上传一张图片，比对 gallery 库，返回最匹配的 top-K 结果
用法:
    # 用训练集做 gallery
    python -m facenet.recognize --model models/facenet_model.pth \\
        --gallery datasets/processed/cropped_train \\
        --probe 我的照片.jpg

    # 用自定义人脸库（每人一个文件夹）
    python -m facenet.recognize --model models/facenet_model.pth \\
        --gallery 我的同事照片/ \\
        --probe 来访客人.jpg \\
        --k 3
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .model import load_model
from .dl_utils import preprocess_single_image, get_test_transforms
from .dl_utils import FaceDataset
from pathlib import PosixPath, WindowsPath
import sys


@torch.no_grad()
def build_gallery(model, gallery_dir: Path, device: str) -> tuple:
    """从 gallery 目录构建人脸库特征矩阵

    Args:
        gallery_dir: 每人一个子目录，目录名 = 人名
    Returns:
        gallery_embs: (N, D) tensor
        gallery_names: [N] 每个人名
    """
    # 把 gallery 当作无增强的 FaceDataset
    dataset = FaceDataset(gallery_dir, transform=get_test_transforms(), augment=False)
    if len(dataset) == 0:
        print(f"[错误] gallery 目录为空: {gallery_dir}")
        sys.exit(1)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    all_embs = []
    all_names = []
    model.eval()

    for imgs, labels in loader:
        imgs = imgs.to(device)
        embs = model.extract_feature(imgs)
        all_embs.append(embs.cpu())
        # 将 label id 转回人名
        batch_names = [dataset.classes[l] for l in labels.numpy()]
        all_names.extend(batch_names)

    gallery_embs = torch.cat(all_embs, dim=0)
    gallery_embs = F.normalize(gallery_embs, p=2, dim=1)

    print(f"[gallery] 已构建 {len(all_names)} 张人脸，共 {len(dataset.classes)} 人")
    return gallery_embs, all_names, dataset.classes


def recognize(model_path: Path, gallery_dir: Path, probe_path: Path, k: int = 5, threshold: float = 0.5):
    """人脸识别主流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(model_path, device=device)
    model.eval()

    # 1. 构建 gallery
    print(f"\n[1/3] 构建 gallery 人脸库: {gallery_dir}")
    gallery_embs, gallery_names, classes = build_gallery(model, gallery_dir, device)

    # 2. 提取 probe 特征
    print(f"\n[2/3] 提取探针图片特征: {probe_path}")
    try:
        probe_tensor = preprocess_single_image(probe_path).to(device)
    except Exception as e:
        print(f"[错误] 无法处理图片 {probe_path}: {e}")
        sys.exit(1)

    probe_emb = model.extract_feature(probe_tensor)
    probe_emb = F.normalize(probe_emb, p=2, dim=1).cpu()

    # 3. 计算余弦相似度并排序
    print(f"\n[3/3] 计算相似度...")
    sim = torch.mm(probe_emb, gallery_embs.t()).squeeze(0)  # (N_gallery,)

    # 按相似度降序排列
    topk_sim, topk_idx = torch.topk(sim, k=min(k, len(sim)))

    # 统计每个人出现次数（统计 top-K 中同一个人出现几次）
    from collections import Counter
    topk_names = [gallery_names[i] for i in topk_idx.tolist()]
    name_votes = Counter(topk_names)

    print(f"\n{'='*55}")
    print(f"  探针图片: {probe_path.name}")
    print(f"{'='*55}")

    # 展示 top-K 结果
    print(f"\n  Top-{k} 匹配结果:")
    print(f"  {'排名':>4} {'姓名':<16} {'相似度':>8} {'图片':<30}")
    print(f"  {'-'*4} {'-'*16} {'-'*8} {'-'*30}")
    for rank, (idx, score) in enumerate(zip(topk_idx.tolist(), topk_sim.tolist()), 1):
        name = gallery_names[idx]
        # 找到对应的图片名（从 gallery 的 samples 中反向查找）
        print(f"  {rank:>4} {name:<16} {score:>7.2%} ")

    # 按投票决定最终身份
    top_person = name_votes.most_common(1)[0]
    avg_sim = np.mean([topk_sim[i].item() for i, n in enumerate(topk_names) if n == top_person[0]])
    max_sim = topk_sim[0].item()

    print(f"\n{'='*55}")
    if max_sim >= threshold:
        print(f"  [结果] 识别为: 【{top_person[0]}】")
        print(f"         置信度: {max_sim:.2%} (最高相似度)")
        print(f"         票数:   {top_person[1]}/{k} (Top-{k} 中该人出现次数)")
    else:
        print(f"  [结果] 未识别 (最高相似度 {max_sim:.2%} < 阈值 {threshold:.0%})")
        print(f"         最接近的候选: {top_person[0]}")
    print(f"{'='*55}")

    # 返回详情
    results = []
    for idx, score in zip(topk_idx.tolist(), topk_sim.tolist()):
        results.append({
            "name": gallery_names[idx],
            "similarity": round(score, 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="FaceNet 人脸识别 — 上传图片，比对 gallery 库")
    parser.add_argument("--model", type=Path, default=Path("models/facenet_model.pth"),
                       help="训练好的模型路径")
    parser.add_argument("--gallery", type=Path, required=True,
                       help="人脸库目录（每人一个子文件夹，文件夹名=人名）")
    parser.add_argument("--probe", type=Path, required=True,
                       help="待识别的人脸图片路径")
    parser.add_argument("--k", type=int, default=5,
                       help="返回 top-K 匹配结果 (默认 5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help='识别阈值，低于此值判定为未知 (默认 0.5)')
    args = parser.parse_args()

    recognize(args.model, args.gallery, args.probe, k=args.k, threshold=args.threshold)


if __name__ == "__main__":
    main()
