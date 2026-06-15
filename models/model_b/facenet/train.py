"""FaceNet 原版训练脚本 — Inception-ResNet-v1 + Triplet Loss + PK 采样

符合原版 FaceNet 论文 (Schroff et al. 2015):
  - PK 采样: 每个 batch 取 P 个人，每人 K 张图
  - Triplet Loss: 带 Batch Hard mining
  - L2 归一化的 embedding
"""

import argparse
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import numpy as np
from .model import FaceNetModel, save_model, load_model
from .dl_losses import TripletLoss, CosFaceLoss
from .dl_utils import FaceDataset, MergedFaceDataset, get_train_transforms


class PKSampler(Sampler):
    """FaceNet PK 采样器: 每个 batch 取 P 个人，每人 K 张图"""

    def __init__(self, labels, P=8, K=8, epoch_size=5000):
        """
        Args:
            labels: [N] 每张图片的类别 id
            P: 每个 batch 的人数量
            K: 每人取 K 张图
            epoch_size: 一个 epoch 产生多少个样本索引
        """
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        self.epoch_size = epoch_size

        # 按类别分组: class_id → [样本索引列表]
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.class_to_indices.setdefault(int(label), []).append(idx)

        # 只保留样本数 >= K 的类
        self.valid_classes = [c for c, idxs in self.class_to_indices.items()
                              if len(idxs) >= K]
        if len(self.valid_classes) < P:
            raise ValueError(
                f"样本数 >= {K} 的类只有 {len(self.valid_classes)} 个，"
                f"但需要 P={P} 个类"
            )

        print(f"[PK采样] P={P}, K={K}, 有效类数={len(self.valid_classes)}")

    def __iter__(self):
        for _ in range(self.epoch_size // (self.P * self.K)):
            # 随机选 P 个类
            chosen_classes = random.sample(self.valid_classes, self.P)
            batch_indices = []
            for cls in chosen_classes:
                # 从该类中随机取 K 张
                indices = self.class_to_indices[cls]
                chosen = random.sample(indices, min(self.K, len(indices)))
                batch_indices.extend(chosen)
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return self.epoch_size


def train(data_dir: Path, output: Path, epochs=60, P=8, K=8, lr=1e-3,
          margin=0.3, pretrained='vggface2',
          loss_type='triplet', extra_data_dirs: list = None, resume: Path = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"损失函数: {loss_type}")

    # ====== 数据加载 ======
    all_dirs = [data_dir] + (extra_data_dirs or [])
    if len(all_dirs) > 1:
        dataset = MergedFaceDataset(all_dirs, transform=get_train_transforms(), augment=True)
    else:
        dataset = FaceDataset(data_dir, transform=get_train_transforms(), augment=True)
    num_classes = len(dataset.classes)
    total_images = len(dataset)
    print(f"检测到 {num_classes} 个类别，共 {total_images} 张图片")

    # ====== 模型初始化 ======
    if resume:
        print(f"[微调] 加载已有模型: {resume}")
        model = load_model(resume, device=device)
        model.to(device)
        model.train()
    else:
        print(f"[从头训练] 使用预训练: {pretrained}")
        model = FaceNetModel(embedding_size=512, pretrained=pretrained).to(device)

    # ====== 损失函数 ======
    if loss_type == 'cosface':
        print(f"CosFace 训练: scale=64.0, margin=0.35, 类别数={num_classes}")
        criterion = CosFaceLoss(
            num_classes=num_classes,
            embedding_size=512,
            scale=64.0, margin=0.35
        ).to(device)

        # CosFace + 标准 DataLoader
        train_loader = DataLoader(
            dataset, batch_size=P * K, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        print(f"Triplet Loss 训练: margin={margin}")
        print(f"PK采样: P={P}(人/批), K={K}(张/人), batch_size={P*K}")
        criterion = TripletLoss(margin=margin).to(device)

        # PK 采样器
        labels = [dataset.samples[i][1] for i in range(total_images)]
        sampler = PKSampler(labels, P=P, K=K, epoch_size=total_images)
        train_loader = DataLoader(
            dataset, batch_size=P * K, sampler=sampler,
            num_workers=0, pin_memory=False, drop_last=True
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                     momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ====== 训练循环 ======
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for imgs, labels_batch in pbar:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            if loss_type == 'cosface':
                embeddings = model(imgs)
                loss = criterion(embeddings, labels_batch)
            else:
                embeddings = model(imgs)
                loss = criterion(embeddings, labels_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        loss_name = 'CosFace' if loss_type == 'cosface' else 'Triplet'
        print(f"Epoch {epoch+1} | {loss_name} Loss: {avg_loss:.4f}")

        # 每 10 个 epoch 保存中间模型
        if (epoch + 1) % 10 == 0:
            intermediate_path = output.parent / f"{output.stem}_ep{epoch+1}.pth"
            save_model(model, intermediate_path, dataset.classes)

    # 保存最终模型
    save_model(model, output, dataset.classes)
    print(f"训练完成！FaceNet 模型已保存: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="训练数据根目录")
    parser.add_argument("--extra-data-dir", type=Path, action='append', default=None,
                       help="额外训练数据目录（可多次使用，如 --extra-data-dir dirA --extra-data-dir dirB）")
    parser.add_argument("--output", type=Path, default=Path("models/facenet_model.pth"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--P", type=int, default=8, help="每个 batch 取 P 个人（triplet 模式）")
    parser.add_argument("--K", type=int, default=8, help="每人取 K 张图 (batch=P*K)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet Loss margin")
    parser.add_argument("--pretrained", action='store_true', default=True,
                       help="使用 ImageNet 预训练权重")
    parser.add_argument("--loss", type=str, default='triplet', choices=['triplet', 'cosface'],
                       help="损失函数: triplet (默认) 或 cosface")
    parser.add_argument("--resume", type=Path, default=None,
                       help="加载已有模型微调（如 models/facenet_model.pth）")
    args = parser.parse_args()

    train(
        args.data_dir, args.output,
        epochs=args.epochs, P=args.P, K=args.K,
        lr=args.lr, margin=args.margin,
        pretrained=args.pretrained,
        loss_type=args.loss,
        extra_data_dirs=args.extra_data_dir,
        resume=args.resume,
    )
