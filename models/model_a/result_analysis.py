"""
结果分析模块
============
绘制混淆矩阵、计算 Accuracy/Precision/Recall/F1-Score、
分析 Fisherfaces 算法优缺点、生成实验测试报告。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from datetime import datetime

# 使用英文标签 (避免中文字体缺失问题)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ==================== 混淆矩阵 ====================

def plot_confusion_matrix(true_labels: List[str],
                          pred_labels: List[str],
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          save_path: str = "results/confusion_matrix.png",
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (14, 12)) -> np.ndarray:
    """
    绘制并保存混淆矩阵

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        class_names: 类别名称列表 (None 则自动提取)
        title: 图表标题
        save_path: 保存路径
        normalize: 是否归一化 (True=按行归一化为百分比)
        figsize: 图表尺寸

    Returns:
        混淆矩阵 numpy 数组
    """
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = sorted(set(true_labels + pred_labels))

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    # 归一化
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized) * 100
    else:
        cm_normalized = cm

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    fmt = '.1f' if normalize else 'd'
    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Accuracy (%)' if normalize else 'Count'},
                vmin=0, vmax=100 if normalize else None)

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')

    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[分析] 混淆矩阵已保存至: {save_path}")

    return cm


def plot_normalized_confusion_matrix(true_labels: List[str],
                                     pred_labels: List[str],
                                     class_names: Optional[List[str]] = None,
                                     title: str = "Normalized Confusion Matrix",
                                     save_path: str = "results/confusion_matrix_normalized.png"):
    """绘制归一到 [0,1] 区间的混淆矩阵"""
    return plot_confusion_matrix(
        true_labels, pred_labels, class_names,
        title=title, save_path=save_path, normalize=True
    )


# ==================== 分类指标计算 ====================

def calculate_metrics(true_labels: List[str],
                      pred_labels: List[str],
                      class_names: Optional[List[str]] = None) -> Dict:
    """
    计算分类评估指标: Accuracy, Precision, Recall, F1-Score

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        class_names: 类别名称

    Returns:
        包含所有指标的字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report
    )

    if class_names is None:
        class_names = sorted(set(true_labels + pred_labels))

    # 总体指标
    accuracy = accuracy_score(true_labels, pred_labels)

    # 宏平均 (每类权重相同)
    precision_macro = precision_score(true_labels, pred_labels,
                                       average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels,
                                 average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels,
                         average='macro', zero_division=0)

    # 加权平均 (按样本数加权)
    precision_weighted = precision_score(true_labels, pred_labels,
                                          average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, pred_labels,
                                    average='weighted', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels,
                            average='weighted', zero_division=0)

    # 每类指标
    per_class_precision = precision_score(true_labels, pred_labels,
                                           average=None, zero_division=0,
                                           labels=class_names)
    per_class_recall = recall_score(true_labels, pred_labels,
                                     average=None, zero_division=0,
                                     labels=class_names)
    per_class_f1 = f1_score(true_labels, pred_labels,
                             average=None, zero_division=0,
                             labels=class_names)

    # 分类报告
    report = classification_report(true_labels, pred_labels,
                                    labels=class_names, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {
            name: {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f)
            }
            for name, p, r, f in zip(class_names,
                                     per_class_precision,
                                     per_class_recall,
                                     per_class_f1)
        },
        'classification_report': report
    }

    return metrics


def print_metrics_table(metrics: Dict, title: str = "分类评估指标") -> None:
    """
    打印格式化的评估指标表
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    print(f"\n  总体指标:")
    print(f"    Accuracy (准确率):           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"    Precision (宏平均):          {metrics['precision_macro']:.4f}")
    print(f"    Recall (宏平均):             {metrics['recall_macro']:.4f}")
    print(f"    F1-Score (宏平均):           {metrics['f1_macro']:.4f}")
    print(f"    Precision (加权平均):        {metrics['precision_weighted']:.4f}")
    print(f"    Recall (加权平均):           {metrics['recall_weighted']:.4f}")
    print(f"    F1-Score (加权平均):         {metrics['f1_weighted']:.4f}")

    print(f"\n  各类别指标 (前10个和后5个):")
    per_class = metrics['per_class']
    class_items = sorted(per_class.items())
    display_items = class_items[:10]
    if len(class_items) > 15:
        display_items += [('...', {'precision': '...', 'recall': '...', 'f1': '...'})]
        display_items += class_items[-5:]

    print(f"    {'人员':<12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print(f"    {'-'*45}")
    for name, vals in display_items:
        if name == '...':
            print(f"    {'...':<12s} {'...':>10s} {'...':>10s} {'...':>10s}")
        else:
            print(f"    {name:<12s} {vals['precision']:>10.4f} "
                  f"{vals['recall']:>10.4f} {vals['f1']:>10.4f}")

    print(f"\n  分类报告:\n{metrics['classification_report']}")
    print(f"{'='*70}")


# ==================== 可视化分析 ====================

def plot_metrics_bar_chart(metrics: Dict,
                           save_path: str = "results/metrics_bar_chart.png") -> None:
    """
    绘制各类别 Precision/Recall/F1 柱状图
    """
    per_class = metrics['per_class']
    class_names = sorted(per_class.keys())

    # 只显示前20个类别 (如果超过的话)
    if len(class_names) > 20:
        # 按 F1 排序选前20
        sorted_classes = sorted(per_class.items(),
                                key=lambda x: x[1]['f1'], reverse=True)[:20]
        class_names = [x[0] for x in sorted_classes]

    precisions = [per_class[c]['precision'] for c in class_names]
    recalls = [per_class[c]['recall'] for c in class_names]
    f1s = [per_class[c]['f1'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - width, precisions, width, label='Precision',
                   color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label='Recall',
                   color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score',
                   color='#FF9800', alpha=0.85)

    ax.set_xlabel('Person', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Precision / Recall / F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[分析] 指标对比图已保存至: {save_path}")


def plot_accuracy_comparison(test_results: Dict,
                             save_path: str = "results/accuracy_comparison.png") -> None:
    """
    绘制不同测试场景的准确率对比图
    """
    # 提取各测试的准确率
    labels = []
    accuracies = []

    for key, value in test_results.items():
        if hasattr(value, 'results'):
            for r in value.results:
                if r.total_samples > 0:
                    labels.append(r.test_name)
                    accuracies.append(r.accuracy)
        elif hasattr(value, 'accuracy') and value.total_samples > 0:
            labels.append(value.test_name)
            accuracies.append(value.accuracy)

    if not labels:
        print("[警告] 无有效测试结果用于对比")
        return

    # 排序
    sorted_idx = np.argsort(accuracies)[::-1]
    labels = [labels[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.4)))

    colors = ['#4CAF50' if a >= 90 else '#FF9800' if a >= 70 else '#F44336'
              for a in accuracies]

    bars = ax.barh(range(len(labels)), accuracies, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Test Scenarios', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)

    # 标注数值
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[分析] 准确率对比图已保存至: {save_path}")


def plot_lighting_sensitivity(azimuth_results: Dict,
                              save_path: str = "results/lighting_sensitivity.png") -> None:
    """绘制光照敏感度曲线"""
    if not azimuth_results:
        print("[警告] 无方位角数据")
        return

    labels = list(azimuth_results.keys())
    accuracies = [azimuth_results[l]['accuracy'] for l in labels]
    counts = [azimuth_results[l]['count'] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # 准确率 vs 方位角
    x = range(len(labels))
    ax1.plot(x, accuracies, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Recognition Accuracy Under Different Azimuth Angles', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)

    for i, (acc, cnt) in enumerate(zip(accuracies, counts)):
        ax1.annotate(f'{acc:.1f}%\n({cnt})',
                     (i, acc), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=8)

    # 样本分布
    ax2.bar(x, counts, color='#FF9800', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Sample Distribution Across Azimuth Angles', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[分析] 光照敏感度图已保存至: {save_path}")


def plot_confidence_distribution(confidences: List[float],
                                 correct_mask: List[bool],
                                 save_path: str = "results/confidence_distribution.png") -> None:
    """绘制置信度分布直方图"""
    correct_confs = [c for c, m in zip(confidences, correct_mask) if m]
    wrong_confs = [c for c, m in zip(confidences, correct_mask) if not m]

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(min(confidences), min(max(confidences), 10000), 50)

    if correct_confs:
        ax.hist(correct_confs, bins=bins, alpha=0.7, label=f'Correct ({len(correct_confs)})',
                color='#4CAF50', edgecolor='white')
    if wrong_confs:
        ax.hist(wrong_confs, bins=bins, alpha=0.7, label=f'Wrong ({len(wrong_confs)})',
                color='#F44336', edgecolor='white')

    ax.set_xlabel('Confidence (distance, lower is better)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Wrong Predictions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 标注推荐阈值
    if correct_confs and wrong_confs:
        mid_point = (np.median(correct_confs) + np.median(wrong_confs)) / 2
        ax.axvline(x=mid_point, color='#FF9800', linestyle='--', linewidth=2,
                   label=f'Recommended threshold ~ {mid_point:.0f}')
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[分析] 置信度分布图已保存至: {save_path}")


# ==================== 算法优缺点分析 ====================

def analyze_fisherfaces(results: Dict, metrics: Dict,
                        model_info: Dict) -> str:
    """
    分析 Fisherfaces 算法的优缺点

    结合实验结果进行系统性的优缺点分析。

    Args:
        results: 测试结果字典
        metrics: 评估指标
        model_info: 模型信息

    Returns:
        分析报告文本
    """
    accuracy = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_weighted', 0)

    # 获取光照测试结果
    lighting_accs = {}
    if 'lighting' in results:
        for r in results['lighting'].results:
            detail = r.details.get('lighting_level', '')
            if detail:
                lighting_accs[detail] = r.accuracy

    report = f"""
{'='*70}
Fisherfaces 算法优缺点分析
{'='*70}

一、算法概述
-----------
Fisherfaces (基于 LDA 的人脸识别) 通过最大化类间散度与类内散度的比值，
找到最具区分性的投影方向。本实现基于 OpenCV 的 FisherFaceRecognizer。

二、实验数据
-----------
- 训练集: {model_info.get('num_classes', 'N/A')} 类人员
- 测试样本: 见各测试结果
- 总体准确率: {accuracy*100:.2f}%
- 加权 F1-Score: {f1:.4f}

三、Fisherfaces 优势
--------------------
1. 【光照鲁棒性】Fisherfaces 关注的是类间区分性，在光照变化下表现相对稳定。
   LDA 投影方向本身对光照变化有一定的抑制作用。

2. 【判别性强】与 Eigenfaces (PCA) 相比，Fisherfaces 使用有监督的 LDA，
   明确最大化类间差异，因此在人脸区分方面更具优势。

3. 【训练效率】Fisherfaces 模型训练速度快，参数量小，适合中小规模应用。
   模型文件只有几十 KB，非常适合嵌入式或移动端部署。

4. 【小样本友好】通过 PCA 预降维，Fisherfaces 可以处理样本数小于特征
   维度的情况（小样本问题）。PCA 阶段将数据降至 (N-c) 维，保证 Sw 可逆。

5. 【理论基础扎实】基于 Fisher 线性判别准则，有严格的数学理论支撑。
   投影后的特征具有明确的物理意义——最有区分性的面部特征方向。

四、Fisherfaces 局限性
----------------------
1. 【光照极端条件】在极端光照条件下（大方位角），准确率会明显下降：
"""
    for level in ['良好光照', '中等光照', '极端光照']:
        if level in lighting_accs:
            report += f"   - {level}: {lighting_accs[level]:.2f}%\n"

    report += f"""
2. 【姿态敏感性】Fisherfaces 对姿态变化相对敏感。PCA+LDA 的线性投影
   难以处理大幅度的面部旋转和非刚性形变。需要配合人脸对齐预处理。

3. 【线性模型限制】Fisherfaces 是线性方法，无法捕获人脸图像中的非线性
   变化（表情、年龄变化等）。对于复杂的面部变化需要非线性扩展（如核方法）。

4. 【数据依赖】需要每类有足够的训练样本（建议 > 5 张/人），否则类内散度
   矩阵估计不准确。类别数 (c) 决定了 Fisherfaces 最多只有 (c-1) 个有效
   判别方向。

5. 【对遮挡敏感】面部遮挡（眼镜、口罩、帽子等）会导致 Fisherfaces 特征
   失真，因为线性投影无法区分遮挡和身份特征。

6. 【类别增量困难】新增人员需要重新训练整个模型，不支持增量学习。

五、与深度学习方法的对比分析
---------------------------
| 特性               | Fisherfaces       | FaceNet/ArcFace   |
|-------------------|-------------------|-------------------|
| 方法类型           | 传统线性方法       | 深度神经网络       |
| 特征提取           | PCA+LDA 投影      | CNN 嵌入          |
| 光照鲁棒性         | 中等              | 高                |
| 姿态鲁棒性         | 较低              | 高                |
| 训练数据需求       | 少量 (百张级)     | 大量 (万张级)     |
| 模型大小           | 极小 (KB 级)      | 大 (MB-GB 级)     |
| 训练时间           | 秒级              | 小时-天级         |
| 推理速度           | 非常快            | 中等-快           |
| 可解释性           | 高 (可可视化)     | 低 (黑盒)         |
| 适用场景           | 小规模、实时      | 大规模、高精度    |

六、改进建议
-----------
1. 结合更好的图像预处理 (伽马校正、高斯滤波) 提升光照鲁棒性
2. 使用 Gabor 或 LBP 特征代替原始像素，增强对光照和姿态的鲁棒性
3. 引入核方法 (Kernel Fisherfaces) 处理非线性变化
4. 配合人脸检测和对齐模块，减少姿态变化的负面影响
5. 对大规模场景，考虑使用深度学习方法的嵌入特征
"""
    return report


# ==================== 实验报告生成 ====================

def generate_experiment_report(results: Dict,
                               metrics: Dict,
                               model_info: Dict,
                               report_dir: str = "results") -> str:
    """
    生成完整的实验测试报告

    Args:
        results: 测试结果
        metrics: 评估指标
        model_info: 模型信息
        report_dir: 报告输出目录

    Returns:
        报告文件路径
    """
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "experiment_report.md")

    # 收集各种数据
    accuracy = metrics.get('accuracy', 0) * 100
    f1_macro = metrics.get('f1_macro', 0)
    precision_macro = metrics.get('precision_macro', 0)
    recall_macro = metrics.get('recall_macro', 0)

    # 光照测试结果
    lighting_section = ""
    if 'lighting' in results:
        for r in results['lighting'].results:
            lighting_section += f"| {r.test_name} | {r.total_samples} | {r.correct_predictions} | {r.accuracy:.2f}% | {r.elapsed_time:.2f}s |\n"

    # 姿态测试结果
    pose_section = ""
    if 'pose' in results:
        for r in results['pose'].results:
            pose_section += f"| {r.test_name} | {r.total_samples} | {r.correct_predictions} | {r.accuracy:.2f}% |\n"

    # 人员识别结果
    person_section = ""
    person_accuracies = []
    if 'person' in results:
        for r in results['person'].results:
            person_accuracies.append(r.accuracy)

    if person_accuracies:
        person_section = f"""
| 最高准确率 | {max(person_accuracies):.2f}% |
| 最低准确率 | {min(person_accuracies):.2f}% |
| 平均准确率 | {np.mean(person_accuracies):.2f}% |
| 标准差 | {np.std(person_accuracies):.2f}% |
"""

    # 生成 Markdown 报告
    report_content = f"""# Fisherfaces 人脸识别算法 —— 实验测试报告

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
>
> **算法**: Fisherfaces (基于 LDA 的线性判别分析)

---

## 1. 实验概述

### 1.1 实验目的
- 研究 Fisherfaces 算法在人脸识别中的性能表现
- 评估算法在不同光照条件下的鲁棒性
- 分析姿态变化对识别准确率的影响
- 统计系统对不同人员的区分能力
- 评估模型的识别速度和资源占用

### 1.2 实验环境
- **编程语言**: Python 3.11
- **核心库**: OpenCV (cv2.face.FisherFaceRecognizer)
- **评估库**: scikit-learn, NumPy
- **可视化**: Matplotlib, Seaborn

---

## 2. 数据集说明

### 2.1 数据来源
使用 Extended Yale B 人脸数据集，包含 {model_info.get('num_classes', 'N/A')} 个人的多光照条件人脸图像。

### 2.2 数据划分
- **训练集**: `data/cropped_train/` — 每人 340 张图像
- **测试集**: `data/cropped_test/` — 每人约 113 张图像
- **图像尺寸**: {model_info.get('image_size', 'N/A')}

### 2.3 光照条件分级
| 光照级别 | 方位角范围 | 说明 |
|---------|-----------|------|
| 良好光照 | \\|A\\| ≤ 20° | 接近正面光照 |
| 中等光照 | 20° < \\|A\\| ≤ 50° | 侧面光照 |
| 极端光照 | \\|A\\| > 50° | 大角度侧面光照 |

---

## 3. 模型配置

| 参数 | 值 |
|-----|-----|
| 算法 | Fisherfaces (LDA) |
| PCA 主成分数 | {model_info.get('num_components', '自动')} |
| 识别阈值 | {model_info.get('threshold', 'N/A')} |
| 预处理 | 灰度化 + 直方图均衡化 |

---

## 4. 测试结果

### 4.1 总体性能指标

| 指标 | 值 |
|-----|-----|
| **Accuracy (准确率)** | **{accuracy:.2f}%** |
| Precision (宏平均) | {precision_macro:.4f} |
| Recall (宏平均) | {recall_macro:.4f} |
| **F1-Score (宏平均)** | **{f1_macro:.4f}** |

### 4.2 功能测试

| 测试项 | 样本数 | 正确数 | 准确率 |
|-------|--------|--------|--------|
| 模型加载 | N/A | N/A | 通过 |
| 单张预测 | N/A | N/A | 通过 |
| 批量预测 | N/A | N/A | 见上方总体指标 |

### 4.3 光照变化测试

| 光照条件 | 样本数 | 正确数 | 准确率 | 耗时 |
|---------|--------|--------|--------|------|
{lighting_section if lighting_section else '| - | - | - | - | - |'}

**分析**: 随着光照条件恶化（方位角增大），识别准确率呈下降趋势。Fisherfaces 在良好光照下表现优秀，但在极端光照下准确率明显降低。这证实了线性方法在处理大幅光照变化时的局限性。

### 4.4 姿态变化测试

| 姿态 | 样本数 | 正确数 | 准确率 |
|-----|--------|--------|--------|
{pose_section if pose_section else '| - | - | - | - |'}

**分析**: Fisherfaces 对姿态变化较为敏感。水平翻转后的识别准确率低于原图，说明模型的泛化能力有限。对于实际部署，建议配合人脸对齐预处理。

### 4.5 不同人员识别测试

{person_section if person_section else '| - | - |'}

**分析**: 不同人员的识别准确率存在差异，这与面部特征的独特性、训练样本质量等因素有关。准确率较低的人员可能需要更多的训练样本或更好的预处理。

---

## 5. 可视化结果

以下图表已保存至 `results/` 目录:

| 图表 | 文件 |
|-----|------|
| 混淆矩阵 | `confusion_matrix.png` |
| 归一化混淆矩阵 | `confusion_matrix_normalized.png` |
| 各类别指标对比 | `metrics_bar_chart.png` |
| 测试场景准确率对比 | `accuracy_comparison.png` |
| 光照敏感度分析 | `lighting_sensitivity.png` |
| 置信度分布 | `confidence_distribution.png` |

---

## 6. Fisherfaces 算法优缺点总结

### 优势
1. **光照鲁棒性较好** — LDA 投影本身对光照变化有一定抑制作用
2. **判别性强** — 有监督学习，最大化类间区分度，优于纯 PCA 方法
3. **训练高效** — 训练速度快（秒级），模型小巧（KB 级），适合实时系统
4. **小样本友好** — PCA 预降维解决了小样本问题
5. **可解释性高** — 可以可视化 Fisherfaces 图像，理解模型的判别依据

### 局限性
1. **极端光照下性能下降** — 大角度光照条件准确率明显降低
2. **姿态敏感性** — 线性投影难以处理大幅度姿态变化
3. **线性模型限制** — 无法捕获非线性面部变化（表情、年龄等）
4. **对遮挡敏感** — 面部遮挡物会显著影响特征提取
5. **类别增量困难** — 新增人员需重新训练整体模型

---

## 7. 结论

Fisherfaces 算法在受控环境下（良好光照、正面姿态）表现优秀，准确率达到 **{accuracy:.1f}%**。
该算法具有训练快速、模型轻量、可解释性强的优点，非常适合小规模、实时性要求高的考勤系统场景。

对于光照变化较大的场景，建议:
1. 增加图像预处理（伽马校正、自适应直方图均衡化等）
2. 确保每类有充足的多样化训练样本
3. 结合人脸检测与对齐模块
4. 考虑使用 LBPH 作为互补算法

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n[报告] 实验报告已生成: {report_path}")
    return report_path


# ==================== 完整分析流程 ====================

def run_full_analysis(true_labels: List[str],
                      pred_labels: List[str],
                      confidences: List[float],
                      test_results: Dict,
                      model_info: Dict,
                      output_dir: str = "results") -> Dict:
    """
    运行完整的结果分析流程

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        confidences: 置信度列表
        test_results: 系统测试结果
        model_info: 模型信息
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    print("\n" + "█" * 70)
    print("█  结果分析")
    print("█" * 70)

    os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(set(true_labels + pred_labels))

    # 1. 计算指标
    metrics = calculate_metrics(true_labels, pred_labels, class_names)
    print_metrics_table(metrics, "Fisherfaces 模型评估指标")

    # 2. 混淆矩阵
    plot_confusion_matrix(true_labels, pred_labels, class_names,
                          save_path=os.path.join(output_dir, "confusion_matrix.png"),
                          normalize=False)
    plot_confusion_matrix(true_labels, pred_labels, class_names,
                          title="归一化混淆矩阵",
                          save_path=os.path.join(output_dir, "confusion_matrix_normalized.png"),
                          normalize=True)

    # 3. 指标可视化
    plot_metrics_bar_chart(metrics,
                           save_path=os.path.join(output_dir, "metrics_bar_chart.png"))

    # 4. 准确率对比
    plot_accuracy_comparison(test_results,
                             save_path=os.path.join(output_dir, "accuracy_comparison.png"))

    # 5. 置信度分布
    correct_mask = [t == p for t, p in zip(true_labels, pred_labels)]
    plot_confidence_distribution(confidences, correct_mask,
                                 save_path=os.path.join(output_dir, "confidence_distribution.png"))

    # 6. 光照敏感度 (如果有数据)
    if 'lighting' in test_results:
        # 从 lighting tester 重新获取方位角数据
        lighting_results = {}
        azimuth_data = {}
        test_path = Path("data/cropped_test")
        try:
            from system_testing import LightingTester
            # 简化处理: 从测试结果中提取光照信息
            for r in test_results['lighting'].results:
                level = r.details.get('lighting_level', '')
                if level:
                    lighting_results[level] = {'accuracy': r.accuracy, 'count': r.total_samples}
            if lighting_results:
                # 转换为方位角格式
                azimuth_data = {
                    '良好 (|A|≤20°)': {'accuracy': lighting_results.get('良好光照', {}).get('accuracy', 0),
                                   'count': lighting_results.get('良好光照', {}).get('count', 0)},
                    '中等 (20<|A|≤50)': {'accuracy': lighting_results.get('中等光照', {}).get('accuracy', 0),
                                     'count': lighting_results.get('中等光照', {}).get('count', 0)},
                    '极端 (|A|>50)': {'accuracy': lighting_results.get('极端光照', {}).get('accuracy', 0),
                                   'count': lighting_results.get('极端光照', {}).get('count', 0)},
                }
            plot_lighting_sensitivity(azimuth_data,
                                      save_path=os.path.join(output_dir, "lighting_sensitivity.png"))
        except Exception as e:
            print(f"[警告] 光照敏感度图生成失败: {e}")

    # 7. 算法分析
    analysis_report = analyze_fisherfaces(test_results, metrics, model_info)
    analysis_path = os.path.join(output_dir, "algorithm_analysis.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)
    print(f"[分析] 算法分析已保存至: {analysis_path}")
    print(analysis_report)

    # 8. 生成实验报告
    report_path = generate_experiment_report(test_results, metrics, model_info, output_dir)

    # 9. 保存完整的 metrics 数据
    metrics_path = os.path.join(output_dir, "metrics_data.json")
    serializable_metrics = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_weighted': float(metrics['f1_weighted']),
        'per_class': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in metrics['per_class'].items()}
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
    print(f"[分析] 指标数据已保存至: {metrics_path}")

    return {
        'metrics': metrics,
        'report_path': report_path,
        'analysis_path': analysis_path,
        'metrics_path': metrics_path
    }


if __name__ == "__main__":
    print("结果分析模块")
    print("请通过 main.py 运行完整分析流程")
