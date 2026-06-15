"""
完整流程: 训练 + 测试 + 分析 (一键运行)
使用优化参数: 50x50 图像, 150 PCA 主成分
"""
import cv2, numpy as np, os, sys, time, pickle
from pathlib import Path

# 确保无缓冲输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 60, flush=True)
print("Fisherfaces 人脸识别系统 - 完整流程", flush=True)
print("=" * 60, flush=True)

# ==================== 配置 ====================
IMAGE_SIZE = (50, 50)
NUM_COMPONENTS = 150
THRESHOLD = 2000.0
TRAIN_DIR = "data/cropped_train"
TEST_DIR = "data/cropped_test"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==================== 阶段 1: 数据加载 ====================
print("\n" + "-" * 50, flush=True)
print("阶段 1/3: 数据加载与预处理", flush=True)
print("-" * 50, flush=True)

from model_training import load_dataset
images, labels, label_map = load_dataset(TRAIN_DIR, IMAGE_SIZE)
reverse_label_map = {v: k for k, v in label_map.items()}

print(f"\n预处理图像 ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})...", flush=True)
processed = []
for img in images:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)
    processed.append(gray)

print(f"完成: {len(processed)} 张图像", flush=True)

# ==================== 阶段 2: 训练 & 测试 ====================
print("\n" + "-" * 50, flush=True)
print("阶段 2/3: 模型训练与系统测试", flush=True)
print("-" * 50, flush=True)

# 训练
print("\n[训练] Fisherfaces 模型...", flush=True)
t0 = time.time()
from fisherfaces_algorithm import FisherfacesModel

model = FisherfacesModel(num_components=NUM_COMPONENTS, threshold=THRESHOLD)
model.label_map = label_map
model.reverse_label_map = reverse_label_map
model.train(images, labels, image_size=IMAGE_SIZE)
train_time = time.time() - t0
print(f"训练耗时: {train_time:.1f}s", flush=True)

# 保存模型
model_path = os.path.join(MODEL_DIR, "fisherfaces_model.yml")
model.save(model_path)

# 保存模型信息
model_info = model.get_model_info()
with open(os.path.join(MODEL_DIR, "model_info.pkl"), 'wb') as f:
    pickle.dump(model_info, f)

# ── 批量预测 ──
print("\n[测试] 标准测试集批量预测...", flush=True)
t0 = time.time()
test_path = Path(TEST_DIR)
all_true, all_pred, all_conf = [], [], []
total, correct = 0, 0

for person_dir in sorted(test_path.iterdir()):
    if not person_dir.is_dir():
        continue
    true_name = person_dir.name
    for ext in ['*.jpg', '*.png']:
        for img_path in person_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                pred_name, conf = model.predict_with_name(img)
            except:
                continue
            all_true.append(true_name)
            all_pred.append(pred_name)
            all_conf.append(conf)
            total += 1
            if pred_name == true_name:
                correct += 1

test_time = time.time() - t0
accuracy = correct / total * 100 if total > 0 else 0
print(f"样本: {total}, 正确: {correct}, 准确率: {accuracy:.2f}%", flush=True)
print(f"测试耗时: {test_time:.1f}s ({total/test_time:.1f} 张/s)", flush=True)

# ── 系统测试 ──
print("\n[测试] 运行系统测试套件...", flush=True)
from system_testing import (
    FunctionalTester, LightingTester, PoseTester,
    PersonIdentificationTester, PerformanceTester,
    TestSuite, TestResult
)

test_results = {}

# 功能测试
func_tester = FunctionalTester(model)
func_suite = func_tester.run_all(TEST_DIR, model_path)
test_results['functional'] = func_suite

# 光照测试
lighting_tester = LightingTester(model)
lighting_suite = lighting_tester.test_lighting_variation(TEST_DIR)
test_results['lighting'] = lighting_suite

# 姿态测试
pose_tester = PoseTester(model)
pose_suite = pose_tester.test_pose_variation(TEST_DIR)
test_results['pose'] = pose_suite

# 人员识别测试
person_tester = PersonIdentificationTester(model)
person_suite = person_tester.test_per_person_accuracy(TEST_DIR)
test_results['person'] = person_suite

# 未知人员拒识
unknown_result = person_tester.test_unknown_person_rejection(TRAIN_DIR, TEST_DIR)
test_results['unknown_rejection'] = unknown_result

# 性能测试
perf_tester = PerformanceTester(model)
perf_speed = perf_tester.test_prediction_speed(TEST_DIR)
perf_size = perf_tester.test_model_size(MODEL_DIR)
test_results['performance_speed'] = perf_speed
test_results['performance_size'] = perf_size

# 保存测试结果
with open(os.path.join(RESULT_DIR, "test_results.pkl"), 'wb') as f:
    pickle.dump(test_results, f)
print(f"\n测试结果已保存至: {RESULT_DIR}/test_results.pkl", flush=True)

# ==================== 阶段 3: 结果分析 ====================
print("\n" + "-" * 50, flush=True)
print("阶段 3/3: 结果分析与报告生成", flush=True)
print("-" * 50, flush=True)

from result_analysis import (
    calculate_metrics, print_metrics_table,
    plot_confusion_matrix, plot_metrics_bar_chart,
    plot_accuracy_comparison, plot_confidence_distribution,
    plot_lighting_sensitivity, analyze_fisherfaces,
    generate_experiment_report, run_full_analysis
)

# 运行完整分析
class_names = sorted(set(all_true + all_pred))
analysis_results = run_full_analysis(
    true_labels=all_true,
    pred_labels=all_pred,
    confidences=all_conf,
    test_results=test_results,
    model_info=model_info,
    output_dir=RESULT_DIR
)

# ==================== 总结 ====================
print("\n" + "=" * 60, flush=True)
print("全部流程完成!", flush=True)
print("=" * 60, flush=True)

print(f"""
输出文件:
  模型:         {MODEL_DIR}/fisherfaces_model.yml
  元数据:       {MODEL_DIR}/fisherfaces_model_meta.pkl
  模型信息:     {MODEL_DIR}/model_info.pkl
  测试结果:     {RESULT_DIR}/test_results.pkl
  实验报告:     {RESULT_DIR}/experiment_report.md
  算法分析:     {RESULT_DIR}/algorithm_analysis.txt
  混淆矩阵:     {RESULT_DIR}/confusion_matrix.png
  归一化混淆:   {RESULT_DIR}/confusion_matrix_normalized.png
  指标对比图:   {RESULT_DIR}/metrics_bar_chart.png
  准确率对比:   {RESULT_DIR}/accuracy_comparison.png
  置信度分布:   {RESULT_DIR}/confidence_distribution.png
  光照敏感度:   {RESULT_DIR}/lighting_sensitivity.png
  指标数据:     {RESULT_DIR}/metrics_data.json
""", flush=True)

# 打印关键指标
metrics = analysis_results.get('metrics', {})
if metrics:
    print(f"""
===== 关键性能指标 =====
总体准确率 (Accuracy):     {metrics['accuracy']*100:.2f}%
宏平均 Precision:          {metrics['precision_macro']:.4f}
宏平均 Recall:             {metrics['recall_macro']:.4f}
宏平均 F1-Score:           {metrics['f1_macro']:.4f}
加权平均 F1-Score:         {metrics['f1_weighted']:.4f}
========================
""", flush=True)
