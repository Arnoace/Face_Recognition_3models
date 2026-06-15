# Fisherfaces 人脸识别考勤系统

基于 OpenCV 实现 Fisherfaces（LDA）人脸识别算法，包含完整的模型训练、系统测试与结果分析流程。

## 算法原理

**Fisherfaces** 由 Belhumeur 等人于 1997 年提出，核心思想是使用 **线性判别分析（LDA）** 寻找最具区分性的人脸特征投影方向：

1. **PCA 降维** —— 将高维人脸图像降至 (N-c) 维，解决类内散度矩阵奇异的小样本问题
2. **LDA 投影** —— 最大化类间散度与类内散度的比值，找到 (c-1) 个 Fisher 判别方向
3. **最近邻分类** —— 将测试图像投影到 Fisherfaces 空间，使用最近邻进行分类

### 与 Eigenfaces 的区别

| 特性 | Eigenfaces (PCA) | Fisherfaces (LDA) |
|------|-----------------|-------------------|
| 学习方式 | 无监督 | 有监督 |
| 优化目标 | 最大方差（重建最优） | 最大类间/类内散度比（判别最优） |
| 光照鲁棒性 | 较差 | 较好 |
| 特征数量 | 最多 N-1 个 | 最多 c-1 个 |

## 项目结构

```
FaceRecognition/
├── fisherfaces_algorithm.py   # Fisherfaces 核心算法（含原理解析类）
├── model_training.py          # 数据加载、模型训练、预测、交叉验证
├── system_testing.py          # 系统测试套件（功能/光照/姿态/人员/性能）
├── result_analysis.py         # 混淆矩阵、分类指标、可视化、实验报告生成
├── main.py                    # 主程序入口（支持命令行参数）
├── run_full_pipeline.py       # 一键运行完整流程
├── requirements.txt           # Python 依赖
├── data/                      # 数据集目录（需自行准备）
│   ├── cropped_train/         #   训练集（按人员分子目录）
│   └── cropped_test/          #   测试集（按人员分子目录）
├── models/                    # 训练好的模型（运行后生成）
└── results/                   # 分析结果与报告（运行后生成）
```

## 环境配置

### 依赖要求

- Python 3.8+
- OpenCV (contrib 版本，包含 face 模块)
- NumPy, scikit-learn, Matplotlib, Seaborn

### 安装

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd FaceRecognition

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

### 数据集准备

使用 **Extended Yale B** 人脸数据集，目录结构如下：

```
data/
├── cropped_train/          # 训练集
│   ├── yaleB11/           # 第 1 个人（340 张）
│   ├── yaleB12/           # 第 2 个人
│   └── ...                # 共 28 人
└── cropped_test/          # 测试集
    ├── yaleB11/           # 第 1 个人（约 113 张）
    └── ...
```

## 使用方法

### 一键运行

```bash
python run_full_pipeline.py
```

### 分步运行

```bash
# 仅训练模型
python main.py --train-only

# 仅测试（需要已有模型）
python main.py --test-only

# 仅分析（需要已有测试结果）
python main.py --analyze-only

# 交叉验证
python main.py --cross-validate
```

### 高级参数

```bash
python main.py \
    --image-size 100 100 \   # 图像尺寸（默认 50 50）
    --num-components 200 \    # PCA 主成分数（默认 150）
    --threshold 1500.0 \      # 识别阈值
    --augment                 # 启用数据增强
```

### 代码示例

```python
from fisherfaces_algorithm import FisherfacesModel
from model_training import train_fisherfaces_model, load_dataset

# 训练模型
model = train_fisherfaces_model(
    data_dir="data/cropped_train",
    image_size=(50, 50),
    num_components=150
)

# 预测
import cv2
img = cv2.imread("test_face.jpg")
name, confidence = model.predict_with_name(img)
print(f"识别结果: {name}, 置信度: {confidence:.2f}")

# 保存 & 加载
model.save("models/my_model.yml")
loaded_model = FisherfacesModel.load("models/my_model.yml")
```

## 测试结果

### 总体性能（28 人，3176 张测试图像）

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)** | **90.81%** |
| 宏平均 Precision | 0.9009 |
| 宏平均 Recall | 0.9071 |
| **宏平均 F1-Score** | **0.9030** |
| 推理速度 | 392.8 张/秒 |

### 光照鲁棒性

| 光照条件 | 方位角 | 准确率 |
|----------|--------|--------|
| 良好光照 | \|A\| ≤ 20° | **97.41%** |
| 中等光照 | 20° < \|A\| ≤ 50° | **92.62%** |
| 极端光照 | \|A\| > 50° | **81.85%** |

> 随着光照条件恶化，准确率呈明显下降趋势，体现了线性方法处理大幅光照变化的局限性。

### Per-Class 最佳/最差

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 🥇 yaleB11 | 0.9946 | 0.9734 | 0.9839 |
| 🥈 yaleB35 | 0.9925 | 0.9433 | 0.9673 |
| 🥉 yaleB28 | 0.9621 | 0.9847 | 0.9732 |
| ... | ... | ... | ... |
| yaleB30 | 0.7255 | 0.8043 | 0.7629 |

## 实验报告

运行后会生成以下可视化结果：

| 图表 | 说明 |
|------|------|
| `confusion_matrix.png` | 28×28 混淆矩阵 |
| `confusion_matrix_normalized.png` | 按行归一化的混淆矩阵 |
| `metrics_bar_chart.png` | 各类别 Precision/Recall/F1 柱状图 |
| `accuracy_comparison.png` | 各测试场景准确率横向对比 |
| `lighting_sensitivity.png` | 不同方位角下的准确率变化曲线 |
| `confidence_distribution.png` | 正确/错误识别的置信度分布 |
| `experiment_report.md` | 完整实验测试报告（Markdown） |

## Fisherfaces 优缺点

### ✅ 优势

- **光照鲁棒性较好** — LDA 投影本身对光照有一定抑制
- **判别性强** — 有监督学习，类间区分度优于 PCA
- **训练高效** — 秒级训练，模型仅 ~30KB
- **推理快速** — 接近 400 FPS
- **小样本友好** — PCA 预降维解决小样本问题
- **可解释性高** — 可可视化 Fisherfaces 图像

### ❌ 局限性

- 极端光照下性能下降明显
- 对姿态变化敏感
- 线性模型无法捕获非线性面部变化
- 类别增量需要重新训练

## 许可证

本项目仅用于学术研究与学习目的。

## 参考文献

- Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.* IEEE TPAMI.
- Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems.* Annals of Eugenics.
