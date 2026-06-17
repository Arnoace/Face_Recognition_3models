"""FaceNet 人脸识别包 — Inception-ResNet-v1 + Triplet Loss"""

__all__ = [
    "FaceNetModel", "load_model", "save_model",
    "FaceDataset", "MergedFaceDataset", "preprocess_single_image",
]

from .model import FaceNetModel, load_model, save_model
from .dl_utils import FaceDataset, MergedFaceDataset, preprocess_single_image

# 可选损失函数（dl_losses.py 在重构中被移除，不影响核心推理功能）
try:
    from .dl_losses import TripletLoss, CosFaceLoss, MagFaceLoss  # noqa
    __all__ += ["TripletLoss", "CosFaceLoss", "MagFaceLoss"]
except ImportError:
    pass
