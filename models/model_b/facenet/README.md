# FaceNet 人脸识别项目（深度学习）

## 简介
本项目为 **FaceNet 风格** 的深度学习人脸识别框架，采用 **ResNet50 + BNNeck** 骨干网络，支持 **CosFace / MagFace** 损失函数。  
保留了原有的传统预处理模块（`processor.py`、`augmentor.py`），确保数据处理流程高度复用和稳定。

## 主要文件
- `model.py`           —— FaceNet 主模型（AttendanceNet）
- `train.py`           —— 训练脚本（支持 CosFace/MagFace）
- `inference.py`       —— 推理与特征提取
- `evaluate.py`        —— 模型评估
- `dl_utils.py`        —— 数据集与预处理（复用 processor.py）
- `dl_losses.py`       —— CosFace 与 MagFace 损失函数
- `processor.py`       —— 现有的人脸检测与预处理（已深度集成）
- `augmentor.py`       —— 数据增强模块

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow tqdm matplotlib