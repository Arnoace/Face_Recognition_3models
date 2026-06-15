"""FaceNet 推理脚本 - 提取特征并进行人脸比对"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .model import load_model
from .dl_utils import preprocess_single_image


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """计算两个特征向量的余弦相似度"""
    return F.cosine_similarity(emb1, emb2, dim=1).item()


def recognize(model_path: Path, probe_path: Path, threshold: float = 0.6):
    """人脸识别主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 FaceNet 模型
    model = load_model(model_path, device=device)
    
    # 预处理图片（复用 processor.py）
    img_tensor = preprocess_single_image(probe_path).to(device)
    
    # 提取特征
    embedding = model.extract_feature(img_tensor)
    
    print(f"特征提取成功，维度: {embedding.shape}，范数: {torch.norm(embedding).item():.4f}")
    return embedding


def main():
    parser = argparse.ArgumentParser(description="FaceNet 人脸识别推理工具")
    parser.add_argument('--model', type=Path, default=Path('models/dl_model_cos.pth'), 
                       help='训练好的模型路径')
    parser.add_argument('--image', type=Path, required=True, help='待识别的图片路径')
    parser.add_argument('--threshold', type=float, default=0.6, help='相似度阈值')
    args = parser.parse_args()
    
    emb = recognize(args.model, args.image, args.threshold)
    print("识别完成！可以使用该 embedding 与注册库进行比对。")


if __name__ == '__main__':
    main()