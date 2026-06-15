"""
模型训练与预测模块
=================
负责数据加载、预处理、模型训练、保存与预测功能。
"""

import cv2
import numpy as np
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from fisherfaces_algorithm import FisherfacesModel


# ==================== 数据加载 ====================

def load_dataset(data_dir: str,
                 image_size: Tuple[int, int] = (100, 100)) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """
    从目录加载人脸数据集

    目录结构应为:
        data_dir/
          person_1/
            image_1.jpg
            image_2.jpg
            ...
          person_2/
            ...

    Args:
        data_dir: 数据集根目录
        image_size: 统一的图像尺寸 (width, height)

    Returns:
        (images, labels, label_map): 图像列表、数字标签列表、标签映射表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    images = []
    labels = []
    label_map = {}

    person_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if len(person_dirs) == 0:
        raise ValueError(f"在 {data_dir} 中未找到任何人员子目录")

    for label_idx, person_dir in enumerate(person_dirs):
        person_name = person_dir.name
        label_map[label_idx] = person_name

        image_files = list(person_dir.glob('*.jpg')) + \
                      list(person_dir.glob('*.png')) + \
                      list(person_dir.glob('*.bmp')) + \
                      list(person_dir.glob('*.pgm'))

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(label_idx)

    print(f"[数据加载] 从 {data_dir} 加载完成:")
    print(f"  - 人员数: {len(person_dirs)}")
    print(f"  - 总图像数: {len(images)}")
    for label_idx, name in sorted(label_map.items()):
        count = labels.count(label_idx)
        print(f"    [{label_idx}] {name}: {count} 张")

    return images, labels, label_map


def load_dataset_by_person(data_dir: str,
                           image_size: Tuple[int, int] = (100, 100)) -> Dict[str, List[np.ndarray]]:
    """
    按人员分组加载数据集

    Args:
        data_dir: 数据集根目录
        image_size: 统一的图像尺寸

    Returns:
        {person_name: [images]} 字典
    """
    data_path = Path(data_dir)
    images_by_person = {}

    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        images = []
        for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
            for img_path in person_dir.glob(ext):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
        if images:
            images_by_person[person_name] = images

    return images_by_person


# ==================== 数据增强 ====================

def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """
    对单张图像进行数据增强

    Args:
        image: 输入图像

    Returns:
        增强后的图像列表 (包含原图)
    """
    augmented = [image]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # 水平翻转
    augmented.append(cv2.flip(image, 1))

    # 轻微旋转 (±5度)
    center = (w // 2, h // 2)
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    return augmented


# ==================== 模型训练 ====================

def train_fisherfaces_model(data_dir: str,
                            image_size: Tuple[int, int] = (100, 100),
                            num_components: int = 0,
                            threshold: float = 2000.0,
                            augment: bool = False) -> FisherfacesModel:
    """
    训练 Fisherfaces 人脸识别模型

    Args:
        data_dir: 训练数据目录
        image_size: 图像尺寸
        num_components: PCA 主成分数 (0=自动)
        threshold: 识别阈值
        augment: 是否使用数据增强

    Returns:
        训练好的 FisherfacesModel
    """
    print("\n" + "=" * 60)
    print("Fisherfaces 模型训练")
    print("=" * 60)

    start_time = time.time()

    # 加载数据
    images, labels, label_map = load_dataset(data_dir, image_size)

    # 数据增强 (可选)
    if augment:
        print("[数据增强] 正在进行数据增强...")
        aug_images, aug_labels = [], []
        for img, lbl in zip(images, labels):
            aug_imgs = augment_image(img)
            aug_images.extend(aug_imgs)
            aug_labels.extend([lbl] * len(aug_imgs))
        images, labels = aug_images, aug_labels
        print(f"  增强后样本数: {len(images)}")

    # 创建模型
    model = FisherfacesModel(
        num_components=num_components,
        threshold=threshold
    )
    model.label_map = label_map
    model.reverse_label_map = {v: k for k, v in label_map.items()}

    # 训练
    model.train(images, labels, image_size=image_size)

    elapsed = time.time() - start_time
    print(f"\n[训练完成] 耗时: {elapsed:.2f} 秒")

    return model


def save_model(model: FisherfacesModel, save_dir: str = "models") -> str:
    """
    保存模型及相关文件

    Args:
        model: 训练好的 FisherfacesModel
        save_dir: 保存目录

    Returns:
        模型文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "fisherfaces_model.yml")
    model.save(model_path)

    # 保存模型信息摘要
    info_path = os.path.join(save_dir, "model_info.pkl")
    with open(info_path, 'wb') as f:
        pickle.dump(model.get_model_info(), f)

    print(f"[保存] 模型信息已保存至: {info_path}")
    return model_path


def load_model(model_dir: str = "models") -> FisherfacesModel:
    """
    加载已保存的模型

    Args:
        model_dir: 模型目录

    Returns:
        FisherfacesModel 实例
    """
    model_path = os.path.join(model_dir, "fisherfaces_model.yml")
    if not os.path.exists(model_path):
        # 尝试 .xml 扩展名
        model_path = os.path.join(model_dir, "fisherfaces_model.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_dir}/fisherfaces_model.yml")

    return FisherfacesModel.load(model_path)


# ==================== 批量预测 ====================

def batch_predict(model: FisherfacesModel,
                  test_dir: str) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    对测试集进行批量预测

    Args:
        model: 训练好的 FisherfacesModel
        test_dir: 测试数据目录

    Returns:
        (true_names, pred_names, confidences, image_paths):
            真实标签、预测标签、置信度、图像路径
    """
    true_names = []
    pred_names = []
    confidences = []
    image_paths = []

    test_path = Path(test_dir)
    total = 0
    correct = 0

    for person_dir in sorted(test_path.iterdir()):
        if not person_dir.is_dir():
            continue
        true_name = person_dir.name

        for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
            for img_path in person_dir.glob(ext):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                try:
                    pred_name, conf = model.predict_with_name(img)
                except Exception as e:
                    print(f"[警告] 预测失败 {img_path}: {e}")
                    continue

                true_names.append(true_name)
                pred_names.append(pred_name)
                confidences.append(conf)
                image_paths.append(str(img_path))

                total += 1
                if pred_name == true_name:
                    correct += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[批量预测] 测试集准确率: {correct}/{total} = {accuracy:.2f}%")

    return true_names, pred_names, confidences, image_paths


# ==================== 交叉验证 ====================

def cross_validate(data_dir: str,
                   n_folds: int = 5,
                   image_size: Tuple[int, int] = (100, 100)) -> Dict[str, float]:
    """
    K 折交叉验证评估模型稳定性

    Args:
        data_dir: 数据集目录
        n_folds: 折数
        image_size: 图像尺寸

    Returns:
        包含各折准确率及平均值的字典
    """
    from sklearn.model_selection import StratifiedKFold

    print("\n" + "=" * 60)
    print(f"K 折交叉验证 (K={n_folds})")
    print("=" * 60)

    # 加载所有数据 (包括 train 和 test)
    images, labels, label_map = load_dataset(data_dir, image_size)

    if len(np.unique(labels)) < 2:
        print("[错误] 交叉验证需要至少2个类别")
        return {}

    # 预处理图像
    processed = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)
        processed.append(gray)

    X = np.array(processed)
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        fold_model = cv2.face.FisherFaceRecognizer_create()
        fold_model.train(list(X_train), np.array(y_train, dtype=np.int32))

        # 预测
        correct = 0
        for i, sample in enumerate(X_val):
            pred_label, _ = fold_model.predict(sample)
            if pred_label == y_val[i]:
                correct += 1

        acc = correct / len(y_val) * 100
        fold_accuracies.append(acc)
        print(f"  Fold {fold}: {len(y_train)} 训练 / {len(y_val)} 验证 → 准确率 {acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n  平均准确率: {mean_acc:.2f}%")
    print(f"  标准差: {std_acc:.2f}%")

    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }


# ==================== 单张预测 ====================

def predict_single_image(model: FisherfacesModel, image_path: str) -> dict:
    """
    预测单张图片的身份

    Args:
        model: 训练好的模型
        image_path: 图片路径

    Returns:
        预测结果字典
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    pred_name, confidence = model.predict_with_name(img)

    return {
        'image_path': image_path,
        'predicted_person': pred_name,
        'confidence': confidence,
        'is_below_threshold': confidence < model.threshold
    }


# ==================== 模块测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("模型训练模块测试")
    print("=" * 60)

    # 测试数据加载
    data_dir = "data/cropped_train"
    if os.path.exists(data_dir):
        images, labels, label_map = load_dataset(data_dir)

        # 训练模型
        model = train_fisherfaces_model(
            data_dir=data_dir,
            image_size=(100, 100),
            num_components=0,
            threshold=2000.0,
            augment=False
        )

        # 保存模型
        save_model(model, save_dir="models")

        # 测试预测
        test_dir = "data/cropped_test"
        if os.path.exists(test_dir):
            true_names, pred_names, confidences, paths = batch_predict(model, test_dir)
    else:
        print(f"[错误] 数据目录不存在: {data_dir}")
        print("请确保数据集已放置在 data/cropped_train/ 目录下")
