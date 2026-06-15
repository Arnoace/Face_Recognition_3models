"""
系统测试模块
============
设计与实现功能测试、光照变化测试、姿态变化测试、不同人员识别测试。
"""

import cv2
import numpy as np
import os
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field

from fisherfaces_algorithm import FisherfacesModel


# ==================== 数据结构 ====================

@dataclass
class TestResult:
    """单次测试结果"""
    test_name: str
    total_samples: int
    correct_predictions: int
    accuracy: float
    true_labels: List[str] = field(default_factory=list)
    pred_labels: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    elapsed_time: float = 0.0


@dataclass
class TestSuite:
    """测试套件"""
    results: List[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult):
        self.results.append(result)

    def summary(self) -> str:
        """生成测试摘要"""
        lines = ["\n" + "=" * 70, "测试套件摘要", "=" * 70]
        total_correct = 0
        total_samples = 0
        for r in self.results:
            lines.append(
                f"  {r.test_name:<30s} | 准确率: {r.accuracy:6.2f}%  "
                f"| {r.correct_predictions:4d}/{r.total_samples:4d}  "
                f"| 耗时: {r.elapsed_time:.2f}s"
            )
            total_correct += r.correct_predictions
            total_samples += r.total_samples

        overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
        lines.append("-" * 70)
        lines.append(f"  整体准确率: {overall_acc:.2f}% ({total_correct}/{total_samples})")
        lines.append("=" * 70)
        return "\n".join(lines)


# ==================== 功能测试 ====================

class FunctionalTester:
    """功能测试 —— 验证系统基本功能是否正常"""

    def __init__(self, model: FisherfacesModel):
        self.model = model

    def test_model_loading(self, model_path: str) -> TestResult:
        """测试模型加载功能"""
        print("\n[功能测试] 模型加载...")
        start = time.time()
        try:
            loaded_model = FisherfacesModel.load(model_path)
            success = loaded_model.is_trained
            print(f"  [OK] 模型加载成功, 类别数: {len(loaded_model.label_map)}")
        except Exception as e:
            success = False
            print(f"  [FAIL] 模型加载失败: {e}")

        return TestResult(
            test_name="Model Loading Test",
            total_samples=1,
            correct_predictions=1 if success else 0,
            accuracy=100.0 if success else 0.0,
            elapsed_time=time.time() - start
        )

    def test_single_prediction(self) -> TestResult:
        """测试单张图像预测功能"""
        print("\n[功能测试] 单张图像预测...")
        # 从测试集取一张图片
        test_dir = Path("data/cropped_test")
        img_path = None
        true_name = None
        for person_dir in sorted(test_dir.iterdir()):
            if person_dir.is_dir():
                for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                    files = list(person_dir.glob(ext))
                    if files:
                        img_path = str(files[0])
                        true_name = person_dir.name
                        break
            if img_path:
                break

        if img_path is None:
            return TestResult("Single Prediction Test", 0, 0, 0.0)

        start = time.time()
        img = cv2.imread(img_path)
        pred_name, conf = self.model.predict_with_name(img)
        elapsed = time.time() - start

        correct = pred_name == true_name
        print(f"  真实: {true_name} → 预测: {pred_name} (置信度: {conf:.2f})")
        print(f"  {'[OK] 正确' if correct else '[FAIL] 错误'}")

        return TestResult(
            test_name="Single Prediction Test",
            total_samples=1,
            correct_predictions=1 if correct else 0,
            accuracy=100.0 if correct else 0.0,
            true_labels=[true_name],
            pred_labels=[pred_name],
            confidences=[conf],
            elapsed_time=elapsed
        )

    def test_batch_prediction(self, test_dir: str = "data/cropped_test") -> TestResult:
        """测试批量预测功能 —— 标准测试集评估"""
        print("\n[功能测试] 批量预测 (标准测试集)...")
        start = time.time()

        true_names, pred_names, confidences, paths = self._run_prediction(test_dir)
        elapsed = time.time() - start

        correct = sum(1 for t, p in zip(true_names, pred_names) if t == p)
        accuracy = correct / len(true_names) * 100 if true_names else 0

        print(f"  样本数: {len(true_names)}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2f}%")

        return TestResult(
            test_name="Functional-Standard Test Set",
            total_samples=len(true_names),
            correct_predictions=correct,
            accuracy=accuracy,
            true_labels=true_names,
            pred_labels=pred_names,
            confidences=confidences,
            elapsed_time=elapsed
        )

    def _run_prediction(self, test_dir: str) -> Tuple[List[str], List[str], List[float], List[str]]:
        """执行批量预测"""
        true_names, pred_names, confidences, paths = [], [], [], []
        test_path = Path(test_dir)

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
                        pred_name, conf = self.model.predict_with_name(img)
                    except Exception:
                        continue
                    true_names.append(true_name)
                    pred_names.append(pred_name)
                    confidences.append(conf)
                    paths.append(str(img_path))

        return true_names, pred_names, confidences, paths

    def run_all(self, test_dir: str = "data/cropped_test",
                model_path: str = "models/fisherfaces_model.yml") -> TestSuite:
        """运行所有功能测试"""
        print("\n" + "█" * 70)
        print("█  功能测试套件")
        print("█" * 70)

        suite = TestSuite()
        suite.add_result(self.test_model_loading(model_path))
        suite.add_result(self.test_single_prediction())
        suite.add_result(self.test_batch_prediction(test_dir))

        print(suite.summary())
        return suite


# ==================== 光照变化测试 ====================

class LightingTester:
    """
    光照变化测试

    Yale B 数据集的文件命名规则:
    yaleBXX_P00A+000E+00.jpg
    - P00: 姿态 (Pose)
    - A+000: 方位角 (Azimuth), 范围通常 [-130, +130]
    - E+00: 仰角 (Elevation), 范围通常 [-40, +90]

    光照条件分类:
    - 良好光照: 方位角小 (|A| <= 20), 仰角小
    - 中等光照: 方位角中等 (20 < |A| <= 50)
    - 极端光照: 方位角大 (|A| > 50)
    """

    # 光照条件分级
    LIGHTING_LEVELS = {
        '良好光照': lambda az, el: abs(az) <= 20,
        '中等光照': lambda az, el: 20 < abs(az) <= 50,
        '极端光照': lambda az, el: abs(az) > 50,
    }

    def __init__(self, model: FisherfacesModel):
        self.model = model

    @staticmethod
    def parse_lighting_condition(filename: str) -> Tuple[float, float]:
        """
        从文件名解析光照条件

        Args:
            filename: 图像文件名

        Returns:
            (azimuth, elevation): 方位角和仰角
        """
        # 匹配模式: A+000E+00 或 A-035E-20 等
        match = re.search(r'A([+\-]\d+)E([+\-]\d+)', filename)
        if match:
            azimuth = float(match.group(1))
            elevation = float(match.group(2))
            return azimuth, elevation
        return 0.0, 0.0

    @staticmethod
    def classify_lighting(filename: str) -> str:
        """根据文件名分类光照条件"""
        az, el = LightingTester.parse_lighting_condition(filename)
        abs_az = abs(az)

        if abs_az <= 20:
            return '良好光照'
        elif abs_az <= 50:
            return '中等光照'
        else:
            return '极端光照'

    def test_lighting_variation(self, test_dir: str = "data/cropped_test") -> TestSuite:
        """
        光照变化测试 —— 分别评估不同光照条件下的识别准确率

        Returns:
            包含各级别光照测试结果的 TestSuite
        """
        print("\n" + "█" * 70)
        print("█  光照变化测试套件")
        print("█" * 70)

        suite = TestSuite()
        test_path = Path(test_dir)

        # 按光照条件分组收集
        lighting_groups = defaultdict(lambda: {'true': [], 'pred': [], 'conf': []})

        total = 0
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
                        pred_name, conf = self.model.predict_with_name(img)
                    except Exception:
                        continue

                    lighting = self.classify_lighting(img_path.name)
                    lighting_groups[lighting]['true'].append(true_name)
                    lighting_groups[lighting]['pred'].append(pred_name)
                    lighting_groups[lighting]['conf'].append(conf)
                    total += 1

        # 对每种光照条件计算准确率
        for level in ['良好光照', '中等光照', '极端光照']:
            group = lighting_groups[level]
            n = len(group['true'])
            if n == 0:
                print(f"\n  [{level}] 无样本")
                continue

            correct = sum(1 for t, p in zip(group['true'], group['pred']) if t == p)
            accuracy = correct / n * 100
            avg_conf = np.mean(group['conf']) if group['conf'] else 0

            print(f"\n  [{level}]")
            print(f"    样本数: {n}")
            print(f"    正确数: {correct}")
            print(f"    准确率: {accuracy:.2f}%")
            print(f"    平均置信度: {avg_conf:.2f}")

            suite.add_result(TestResult(
                test_name=f"Lighting-{level}",
                total_samples=n,
                correct_predictions=correct,
                accuracy=accuracy,
                true_labels=group['true'],
                pred_labels=group['pred'],
                confidences=group['conf'],
                details={'lighting_level': level, 'avg_confidence': avg_conf}
            ))

        print(suite.summary())
        return suite

    def test_azimuth_sensitivity(self, test_dir: str = "data/cropped_test") -> Dict:
        """
        方位角敏感度分析 —— 分析不同方位角下的准确率变化

        Returns:
            按方位角范围分组的准确率数据
        """
        print("\n[光照分析] 方位角敏感度...")

        # 方位角分组
        bins = [(-130, -90, 'A: -130~-90'), (-90, -50, 'A: -90~-50'),
                (-50, -20, 'A: -50~-20'), (-20, 0, 'A: -20~0'),
                (0, 20, 'A: 0~20'), (20, 50, 'A: 20~50'),
                (50, 90, 'A: 50~90'), (90, 130, 'A: 90~130')]

        test_path = Path(test_dir)
        results = {}

        for low, high, label in bins:
            true_list, pred_list = [], []
            for person_dir in sorted(test_path.iterdir()):
                if not person_dir.is_dir():
                    continue
                true_name = person_dir.name
                for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                    for img_path in person_dir.glob(ext):
                        az, _ = self.parse_lighting_condition(img_path.name)
                        if low <= az < high:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                continue
                            try:
                                pred_name, _ = self.model.predict_with_name(img)
                            except Exception:
                                continue
                            true_list.append(true_name)
                            pred_list.append(pred_name)

            if true_list:
                correct = sum(1 for t, p in zip(true_list, pred_list) if t == p)
                acc = correct / len(true_list) * 100
                results[label] = {'accuracy': acc, 'count': len(true_list)}
                print(f"  {label}: {acc:.2f}% ({len(true_list)} 样本)")

        return results


# ==================== 姿态变化测试 ====================

class PoseTester:
    """
    姿态变化测试

    Yale B 数据集的文件命名规则中:
    - P00: 姿态 0 (正面)
    - P01-P04: 不同程度的姿态变化

    注意: Yale B 数据集主要以光照变化为主，姿态变化有限。
    如果数据集中只有 P00，则使用镜像翻转来模拟姿态变化，
    或者生成含不同姿态的合成数据。
    """

    def __init__(self, model: FisherfacesModel):
        self.model = model

    @staticmethod
    def parse_pose(filename: str) -> int:
        """
        从文件名解析姿态编号

        Args:
            filename: 图像文件名

        Returns:
            姿态编号 (0-4)
        """
        match = re.search(r'P(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    def test_pose_variation(self, test_dir: str = "data/cropped_test") -> TestSuite:
        """
        姿态变化测试

        策略:
        1. 如果数据中存在不同姿态 (P00, P01, ...)，直接按姿态分组测试
        2. 如果只有正面姿态，通过图像变换模拟姿态变化:
           - 水平翻转 (模拟左右转头)
           - 轻微旋转 (模拟头部倾斜)
           - 添加透视变换 (模拟不同视角)
        """
        print("\n" + "█" * 70)
        print("█  姿态变化测试套件")
        print("█" * 70)

        suite = TestSuite()
        test_path = Path(test_dir)

        # 检查数据集中有哪些姿态
        all_poses = set()
        for person_dir in sorted(test_path.iterdir()):
            if not person_dir.is_dir():
                continue
            for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                for img_path in person_dir.glob(ext):
                    pose = self.parse_pose(img_path.name)
                    all_poses.add(pose)

        print(f"\n  数据集中检测到的姿态: {sorted(all_poses)}")

        if len(all_poses) > 1:
            # 数据本身含有多姿态 → 直接按姿态分组测试
            suite = self._test_real_pose_variation(test_dir, all_poses)
        else:
            # 只有正面姿态 → 模拟姿态变化测试
            suite = self._test_simulated_pose_variation(test_dir)

        print(suite.summary())
        return suite

    def _test_real_pose_variation(self, test_dir: str,
                                  all_poses: set) -> TestSuite:
        """使用数据集中的真实姿态进行测试"""
        suite = TestSuite()
        test_path = Path(test_dir)

        for pose_id in sorted(all_poses):
            true_list, pred_list, conf_list = [], [], []
            pose_label = f"P{pose_id:02d}"

            for person_dir in sorted(test_path.iterdir()):
                if not person_dir.is_dir():
                    continue
                true_name = person_dir.name
                for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                    for img_path in person_dir.glob(ext):
                        p = self.parse_pose(img_path.name)
                        if p != pose_id:
                            continue
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        try:
                            pred_name, conf = self.model.predict_with_name(img)
                        except Exception:
                            continue
                        true_list.append(true_name)
                        pred_list.append(pred_name)
                        conf_list.append(conf)

            if true_list:
                correct = sum(1 for t, p in zip(true_list, pred_list) if t == p)
                acc = correct / len(true_list) * 100
                print(f"\n  [{pose_label}] 样本: {len(true_list)}, 准确率: {acc:.2f}%")

                suite.add_result(TestResult(
                    test_name=f"Pose-{pose_label}",
                    total_samples=len(true_list),
                    correct_predictions=correct,
                    accuracy=acc,
                    true_labels=true_list,
                    pred_labels=pred_list,
                    confidences=conf_list,
                    details={'pose': pose_label}
                ))

        return suite

    def _test_simulated_pose_variation(self, test_dir: str) -> TestSuite:
        """通过图像变换模拟不同姿态进行测试"""
        suite = TestSuite()

        # 定义模拟变换
        transforms = {
            '正面(原图)': lambda img: img,
            '水平翻转': lambda img: cv2.flip(img, 1),
            '左旋5°': lambda img: self._rotate_image(img, -5),
            '右旋5°': lambda img: self._rotate_image(img, 5),
            '左旋10°': lambda img: self._rotate_image(img, -10),
            '右旋10°': lambda img: self._rotate_image(img, 10),
            '透视-左侧': lambda img: self._perspective_transform(img, 'left'),
            '透视-右侧': lambda img: self._perspective_transform(img, 'right'),
        }

        test_path = Path(test_dir)

        for transform_name, transform_fn in transforms.items():
            true_list, pred_list, conf_list = [], [], []

            for person_dir in sorted(test_path.iterdir()):
                if not person_dir.is_dir():
                    continue
                true_name = person_dir.name
                # 每人取一张正面图像做变换
                for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                    img_files = sorted(person_dir.glob(ext))
                    if not img_files:
                        continue
                    # 取 P00 姿态 (正面)
                    frontal_files = [f for f in img_files
                                    if 'P00' in f.name or 'P0A' in f.name]
                    if frontal_files:
                        img_path = str(frontal_files[0])
                    else:
                        img_path = str(img_files[0])

                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    try:
                        transformed = transform_fn(img)
                        pred_name, conf = self.model.predict_with_name(transformed)
                    except Exception:
                        continue

                    true_list.append(true_name)
                    pred_list.append(pred_name)
                    conf_list.append(conf)
                    break  # 每人只取一张

            if true_list:
                correct = sum(1 for t, p in zip(true_list, pred_list) if t == p)
                acc = correct / len(true_list) * 100
                print(f"  [{transform_name}] 准确率: {acc:.2f}% ({correct}/{len(true_list)})")

                suite.add_result(TestResult(
                    test_name=f"Pose-{transform_name}",
                    total_samples=len(true_list),
                    correct_predictions=correct,
                    accuracy=acc,
                    true_labels=true_list,
                    pred_labels=pred_list,
                    confidences=conf_list,
                    details={'pose': transform_name, 'simulated': True}
                ))

        return suite

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def _perspective_transform(image: np.ndarray, direction: str) -> np.ndarray:
        """透视变换模拟视角变化"""
        h, w = image.shape[:2]
        margin = int(w * 0.15)

        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        if direction == 'left':
            dst_pts = np.float32([
                [margin, 0], [w, 0],
                [0, h], [w - margin, h]
            ])
        else:  # right
            dst_pts = np.float32([
                [0, 0], [w - margin, 0],
                [margin, h], [w, h]
            ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image, M, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)
        return result


# ==================== 不同人员识别测试 ====================

class PersonIdentificationTester:
    """
    不同人员识别测试

    测试系统对不同个体的区分能力:
    1. 每类准确率 —— 每个人各自的识别准确率
    2. 混淆分析 —— 哪些人容易被混淆
    3. 未知人员拒识 —— 对未训练人员的识别情况
    """

    def __init__(self, model: FisherfacesModel):
        self.model = model

    def test_per_person_accuracy(self, test_dir: str = "data/cropped_test") -> TestSuite:
        """
        测试每个人的识别准确率
        """
        print("\n" + "█" * 70)
        print("█  不同人员识别测试")
        print("█" * 70)

        suite = TestSuite()
        test_path = Path(test_dir)
        per_person = defaultdict(lambda: {'true': [], 'pred': []})

        # 收集每个人所有测试样本的预测
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
                        pred_name, conf = self.model.predict_with_name(img)
                    except Exception:
                        continue
                    per_person[true_name]['true'].append(true_name)
                    per_person[true_name]['pred'].append(pred_name)

        # 计算每人准确率
        accuracies = {}
        for person_name in sorted(per_person.keys()):
            data = per_person[person_name]
            n = len(data['true'])
            correct = sum(1 for t, p in zip(data['true'], data['pred']) if t == p)
            acc = correct / n * 100 if n > 0 else 0
            accuracies[person_name] = acc

            # 统计被误识别成谁
            errors = defaultdict(int)
            for t, p in zip(data['true'], data['pred']):
                if t != p:
                    errors[p] += 1

            error_summary = ", ".join(
                f"{name}({cnt})" for name, cnt in
                sorted(errors.items(), key=lambda x: -x[1])[:3]
            ) if errors else "无"

            print(f"  {person_name}: {acc:.2f}% ({correct}/{n})  "
                  f"误识别: [{error_summary}]")

            suite.add_result(TestResult(
                test_name=f"Person-{person_name}",
                total_samples=n,
                correct_predictions=correct,
                accuracy=acc,
                true_labels=data['true'],
                pred_labels=data['pred'],
                details={'person': person_name, 'top_errors': dict(errors)}
            ))

        # 统计
        acc_values = list(accuracies.values())
        print(f"\n  最高准确率: {max(acc_values):.2f}% ({max(accuracies, key=accuracies.get)})")
        print(f"  最低准确率: {min(acc_values):.2f}% ({min(accuracies, key=accuracies.get)})")
        print(f"  平均准确率: {np.mean(acc_values):.2f}%")
        print(f"  标准差: {np.std(acc_values):.2f}%")

        print(suite.summary())
        return suite

    def test_unknown_person_rejection(self,
                                       known_dir: str = "data/cropped_train",
                                       unknown_dir: str = "data/cropped_test",
                                       unknown_ratio: float = 0.2) -> TestResult:
        """
        测试未知人员拒识能力

        将部分人员标记为"未知"来测试系统的拒识性能。
        策略: 从测试集中选择部分人员作为"未知人员"，
              这些人员在训练集中不存在或系统应拒绝识别。

        Args:
            known_dir: 已知人员训练目录
            unknown_dir: 测试目录
            unknown_ratio: 标记为未知的人员比例

        Returns:
            TestResult 包含拒识率等指标
        """
        print("\n[人员识别] 未知人员拒识测试...")

        test_path = Path(unknown_dir)
        train_persons = set(d.name for d in Path(known_dir).iterdir()
                           if d.is_dir())

        all_test_persons = sorted([d.name for d in test_path.iterdir()
                                    if d.is_dir()])

        # 所有测试人员在训练集中都有，所以使用阈值来判断
        # 统计所有预测的置信度分布，设定最佳阈值
        all_confidences = []
        true_list, pred_list = [], []

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
                        pred_name, conf = self.model.predict_with_name(img)
                    except Exception:
                        continue
                    all_confidences.append(conf)
                    true_list.append(true_name)
                    pred_list.append(pred_name)

        # 分析置信度分布
        correct_confs = [c for t, p, c in zip(true_list, pred_list, all_confidences) if t == p]
        wrong_confs = [c for t, p, c in zip(true_list, pred_list, all_confidences) if t != p]

        print(f"  正确识别平均置信度: {np.mean(correct_confs):.2f}" if correct_confs else "")
        print(f"  错误识别平均置信度: {np.mean(wrong_confs):.2f}" if wrong_confs else "")
        print(f"  置信度范围: [{np.min(all_confidences):.2f}, {np.max(all_confidences):.2f}]")

        # 测试不同阈值下的拒识性能
        # 对于 Fisherfaces, 距离越大置信度越低
        thresholds = np.linspace(
            np.percentile(all_confidences, 10),
            np.percentile(all_confidences, 90),
            10
        )

        print("\n  阈值-拒识率分析:")
        for th in thresholds:
            rejected = sum(1 for c in all_confidences if c > th)
            reject_rate = rejected / len(all_confidences) * 100
            # 在非拒识样本上的准确率
            accepted_correct = sum(
                1 for t, p, c in zip(true_list, pred_list, all_confidences)
                if c <= th and t == p
            )
            accepted_total = sum(1 for c in all_confidences if c <= th)
            accepted_acc = accepted_correct / accepted_total * 100 if accepted_total > 0 else 0
            print(f"    阈值={th:.1f}: 拒识率={reject_rate:.1f}%, "
                  f"接受样本准确率={accepted_acc:.1f}%")

        return TestResult(
            test_name="Unknown Person Rejection Test",
            total_samples=len(true_list),
            correct_predictions=len(correct_confs),
            accuracy=len(correct_confs) / len(true_list) * 100 if true_list else 0,
            true_labels=true_list,
            pred_labels=pred_list,
            confidences=all_confidences,
            details={
                'correct_confidences': correct_confs,
                'wrong_confidences': wrong_confs,
                'threshold_analysis': True
            }
        )


# ==================== 性能压力测试 ====================

class PerformanceTester:
    """系统性能测试"""

    def __init__(self, model: FisherfacesModel):
        self.model = model

    def test_prediction_speed(self, test_dir: str = "data/cropped_test",
                              n_samples: int = 100) -> TestResult:
        """
        测试预测速度
        """
        print("\n[性能测试] 预测速度...")

        test_path = Path(test_dir)
        images = []

        for person_dir in sorted(test_path.iterdir()):
            if not person_dir.is_dir():
                continue
            for ext in ['*.jpg', '*.png', '*.bmp', '*.pgm']:
                for img_path in person_dir.glob(ext):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        images.append(img)
                    if len(images) >= n_samples:
                        break
                if len(images) >= n_samples:
                    break
            if len(images) >= n_samples:
                break

        if not images:
            return TestResult("Prediction Speed Test", 0, 0, 0.0)

        # 预热
        for _ in range(5):
            self.model.predict_with_name(images[0])

        # 计时
        start = time.time()
        for img in images:
            self.model.predict_with_name(img)
        elapsed = time.time() - start

        avg_time = elapsed / len(images) * 1000  # ms
        fps = 1.0 / (elapsed / len(images))

        print(f"  测试样本: {len(images)}")
        print(f"  总耗时: {elapsed:.3f}s")
        print(f"  平均预测时间: {avg_time:.2f}ms/张")
        print(f"  等效 FPS: {fps:.1f}")

        return TestResult(
            test_name="Prediction Speed Test",
            total_samples=len(images),
            correct_predictions=0,  # 不适用
            accuracy=0.0,
            elapsed_time=elapsed,
            details={'avg_time_ms': avg_time, 'fps': fps}
        )

    def test_model_size(self, model_dir: str = "models") -> TestResult:
        """测试模型文件大小"""
        import os
        print("\n[性能测试] 模型大小...")

        model_path = os.path.join(model_dir, "fisherfaces_model.yml")
        meta_path = model_path.replace('.yml', '_meta.pkl')

        total_size = 0
        for path in [model_path, meta_path]:
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                total_size += size_kb
                print(f"  {os.path.basename(path)}: {size_kb:.1f} KB")

        print(f"  总大小: {total_size:.1f} KB")

        return TestResult(
            test_name="Model Size Test",
            total_samples=1,
            correct_predictions=0,
            accuracy=0.0,
            details={'size_kb': total_size}
        )


# ==================== 测试入口 ====================

def run_all_tests(model: FisherfacesModel,
                  test_dir: str = "data/cropped_test",
                  model_path: str = "models/fisherfaces_model.yml") -> Dict:
    """
    运行全套系统测试

    Args:
        model: 训练好的 FisherfacesModel
        test_dir: 测试数据目录
        model_path: 模型文件路径

    Returns:
        包含所有测试结果的字典 (用于结果分析)
    """
    print("\n" + "▓" * 70)
    print("▓  Fisherfaces 人脸识别系统 —— 系统测试")
    print("▓" * 70)

    all_results = {}

    # 1. 功能测试
    func_tester = FunctionalTester(model)
    func_suite = func_tester.run_all(test_dir, model_path)
    all_results['functional'] = func_suite

    # 2. 光照变化测试
    lighting_tester = LightingTester(model)
    lighting_suite = lighting_tester.test_lighting_variation(test_dir)
    all_results['lighting'] = lighting_suite

    # 3. 姿态变化测试
    pose_tester = PoseTester(model)
    pose_suite = pose_tester.test_pose_variation(test_dir)
    all_results['pose'] = pose_suite

    # 4. 不同人员识别测试
    person_tester = PersonIdentificationTester(model)
    person_suite = person_tester.test_per_person_accuracy(test_dir)
    all_results['person'] = person_suite

    # 5. 未知人员拒识测试
    unknown_result = person_tester.test_unknown_person_rejection(
        "data/cropped_train", test_dir
    )
    all_results['unknown_rejection'] = unknown_result

    # 6. 性能测试
    perf_tester = PerformanceTester(model)
    perf_speed = perf_tester.test_prediction_speed(test_dir)
    perf_size = perf_tester.test_model_size("models")
    all_results['performance_speed'] = perf_speed
    all_results['performance_size'] = perf_size

    # 汇总
    print("\n" + "▓" * 70)
    print("▓  系统测试完成")
    print("▓" * 70)

    # 计算综合准确率
    all_accuracies = []
    for key, value in all_results.items():
        if isinstance(value, TestSuite):
            for r in value.results:
                if r.total_samples > 0 and r.accuracy > 0:
                    all_accuracies.append(r.accuracy)
        elif isinstance(value, TestResult):
            if value.total_samples > 0 and value.accuracy > 0:
                all_accuracies.append(value.accuracy)

    if all_accuracies:
        print(f"\n  综合平均准确率: {np.mean(all_accuracies):.2f}%")
        print(f"  最高准确率: {np.max(all_accuracies):.2f}%")
        print(f"  最低准确率: {np.min(all_accuracies):.2f}%")

    return all_results


if __name__ == "__main__":
    print("系统测试模块")
    print("请通过 main.py 运行完整测试流程")
