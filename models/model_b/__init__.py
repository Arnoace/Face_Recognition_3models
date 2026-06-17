"""FaceNet 人脸识别模型集成

提供 FaceNetModel 类（implements BaseFaceModel），
用于在多模型人脸识别系统中注册使用。

支持两个模型:
  - facenet_pretrained: VGGFace2 预训练 InceptionResNetV1（默认）
  - facenet_v6: 本地训练的 GoogLeNet FaceNet
"""

from .interface import FaceNetModel

__all__ = ["FaceNetModel"]
