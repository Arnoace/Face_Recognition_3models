"""
Fisherfaces 人脸识别考勤系统 —— 主程序入口
===========================================

功能流程:
1. 数据加载与预处理
2. Fisherfaces 模型训练
3. 模型保存
4. 系统测试 (功能/光照/姿态/不同人员)
5. 结果分析与可视化
6. 实验报告生成

用法:
    python main.py                    # 运行完整流程
    python main.py --train-only       # 仅训练模型
    python main.py --test-only        # 仅测试 (需已有模型)
    python main.py --analyze-only     # 仅分析 (需已有测试结果)
    python main.py --cross-validate   # 运行交叉验证
"""

import os
import sys
import argparse
import time
import pickle
from pathlib import Path

# 导入各模块
from fisherfaces_algorithm import FisherfacesModel, FisherfacesAlgorithm
from model_training import (
    load_dataset, train_fisherfaces_model, save_model,
    load_model, batch_predict, cross_validate
)
from system_testing import (
    FunctionalTester, LightingTester, PoseTester,
    PersonIdentificationTester, PerformanceTester,
    run_all_tests
)
from result_analysis import (
    calculate_metrics, print_metrics_table,
    plot_confusion_matrix, plot_metrics_bar_chart,
    plot_accuracy_comparison, plot_confidence_distribution,
    plot_lighting_sensitivity, analyze_fisherfaces,
    generate_experiment_report, run_full_analysis
)


# ==================== 配置 ====================

CONFIG = {
    'train_dir': 'data/cropped_train',
    'test_dir': 'data/cropped_test',
    'model_dir': 'models',
    'result_dir': 'results',
    'image_size': (50, 50),
    'num_components': 150,      # PCA 主成分数
    'threshold': 2000.0,        # 识别阈值
    'augment': False,           # 是否数据增强
    'n_folds': 5,               # 交叉验证折数
}


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║       Fisherfaces 人脸识别考勤系统                        ║
    ║       Fisherfaces Algorithm for Face Recognition          ║
    ║       Based on LDA (Linear Discriminant Analysis)         ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_train_pipeline(config: dict) -> FisherfacesModel:
    """
    训练流程: 加载数据 → 训练模型 → 保存模型

    Args:
        config: 配置字典

    Returns:
        训练好的模型
    """
    print("\n" + "━" * 60)
    print("  阶段 1/4: 模型训练")
    print("━" * 60)

    if not os.path.exists(config['train_dir']):
        print(f"[错误] 训练数据目录不存在: {config['train_dir']}")
        sys.exit(1)

    model = train_fisherfaces_model(
        data_dir=config['train_dir'],
        image_size=config['image_size'],
        num_components=config['num_components'],
        threshold=config['threshold'],
        augment=config['augment']
    )

    save_model(model, save_dir=config['model_dir'])

    # 保存模型信息供后续使用
    model_info = model.get_model_info()
    info_path = os.path.join(config['model_dir'], 'model_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)

    return model


def run_test_pipeline(model: FisherfacesModel, config: dict) -> dict:
    """
    测试流程: 运行完整的系统测试套件

    Args:
        model: 训练好的模型
        config: 配置字典

    Returns:
        测试结果字典
    """
    print("\n" + "━" * 60)
    print("  阶段 2/4: 系统测试")
    print("━" * 60)

    test_results = run_all_tests(
        model,
        test_dir=config['test_dir'],
        model_path=os.path.join(config['model_dir'], 'fisherfaces_model.yml')
    )

    # 保存测试结果
    os.makedirs(config['result_dir'], exist_ok=True)
    results_path = os.path.join(config['result_dir'], 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"\n[保存] 测试结果已保存至: {results_path}")

    return test_results


def run_analysis_pipeline(true_labels: list, pred_labels: list,
                          confidences: list, test_results: dict,
                          model_info: dict, config: dict) -> dict:
    """
    分析流程: 计算指标 → 可视化 → 生成报告

    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        confidences: 置信度
        test_results: 测试结果
        model_info: 模型信息
        config: 配置字典

    Returns:
        分析结果字典
    """
    print("\n" + "━" * 60)
    print("  阶段 3/4: 结果分析")
    print("━" * 60)

    analysis_results = run_full_analysis(
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        test_results=test_results,
        model_info=model_info,
        output_dir=config['result_dir']
    )

    return analysis_results


def run_cross_validation(config: dict):
    """
    运行交叉验证

    Args:
        config: 配置字典
    """
    print("\n" + "━" * 60)
    print("  交叉验证")
    print("━" * 60)

    # 使用完整的 cropped 数据集 (合并 train 和 test)
    data_dir = config['train_dir'].replace('_train', '')
    if not os.path.exists(data_dir):
        # 如果 cropped 不存在, 使用 train
        data_dir = config['train_dir']

    cv_results = cross_validate(
        data_dir=data_dir,
        n_folds=config['n_folds'],
        image_size=config['image_size']
    )

    # 保存交叉验证结果
    os.makedirs(config['result_dir'], exist_ok=True)
    cv_path = os.path.join(config['result_dir'], 'cross_validation.pkl')
    with open(cv_path, 'wb') as f:
        pickle.dump(cv_results, f)
    print(f"\n[保存] 交叉验证结果已保存至: {cv_path}")

    return cv_results


def run_full_pipeline(config: dict):
    """
    运行完整流程: 训练 → 测试 → 分析 → 报告
    """
    total_start = time.time()

    # ── 阶段 1: 训练 ──
    model = run_train_pipeline(config)
    model_info = model.get_model_info()

    # ── 阶段 2: 测试 ──
    test_results = run_test_pipeline(model, config)

    # 从功能测试结果中提取标签数据供分析使用
    # (使用标准测试集的结果)
    true_labels, pred_labels, confidences = [], [], []
    if 'functional' in test_results:
        for r in test_results['functional'].results:
            if r.test_name == 'Functional-Standard Test Set' and r.total_samples > 0:
                true_labels = r.true_labels
                pred_labels = r.pred_labels
                confidences = r.confidences
                break

    # 如果功能测试中没有找到，尝试从人员识别测试获取
    if not true_labels and 'person' in test_results:
        for r in test_results['person'].results:
            true_labels.extend(r.true_labels)
            pred_labels.extend(r.pred_labels)
            confidences.extend(r.confidences)

    # ── 阶段 3: 分析 ──
    if true_labels and pred_labels:
        analysis_results = run_analysis_pipeline(
            true_labels, pred_labels, confidences,
            test_results, model_info, config
        )
    else:
        print("[警告] 没有找到有效的测试标签数据，跳过分析阶段")
        analysis_results = {}

    # ── 阶段 4: 总结 ──
    total_elapsed = time.time() - total_start

    print("\n" + "█" * 70)
    print("█  全部流程完成!")
    print(f"█  总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.2f} 分钟)")
    print("█" * 70)

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  输出文件:                                               ║
    ║    模型:     {config['model_dir']}/fisherfaces_model.yml
    ║    元数据:   {config['model_dir']}/fisherfaces_model_meta.pkl
    ║    结果:     {config['result_dir']}/test_results.pkl
    ║    报告:     {config['result_dir']}/experiment_report.md
    ║    混淆矩阵: {config['result_dir']}/confusion_matrix.png
    ║    指标图:   {config['result_dir']}/metrics_bar_chart.png
    ║    对比图:   {config['result_dir']}/accuracy_comparison.png
    ╚══════════════════════════════════════════════════════════╝
    """)

    return model, test_results, analysis_results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='Fisherfaces 人脸识别考勤系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    运行完整流程
  python main.py --train-only       仅训练模型
  python main.py --test-only        仅测试 (需已有模型)
  python main.py --cross-validate   交叉验证
  python main.py --image-size 120 120  自定义图像尺寸
  python main.py --augment          启用数据增强
        """
    )

    parser.add_argument('--train-only', action='store_true',
                        help='仅训练并保存模型')
    parser.add_argument('--test-only', action='store_true',
                        help='仅运行测试 (需要已有模型)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='仅运行分析 (需要已有测试结果)')
    parser.add_argument('--cross-validate', action='store_true',
                        help='运行 K 折交叉验证')
    parser.add_argument('--image-size', type=int, nargs=2, default=[100, 100],
                        metavar=('W', 'H'), help='图像尺寸 (默认: 100 100)')
    parser.add_argument('--threshold', type=float, default=2000.0,
                        help='识别阈值 (默认: 2000.0)')
    parser.add_argument('--num-components', type=int, default=0,
                        help='PCA 主成分数 (默认: 0=自动)')
    parser.add_argument('--augment', action='store_true',
                        help='启用数据增强')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--train-dir', type=str, default='data/cropped_train',
                        help='训练数据目录')
    parser.add_argument('--test-dir', type=str, default='data/cropped_test',
                        help='测试数据目录')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='结果输出目录')

    args = parser.parse_args()

    # 更新配置
    config = CONFIG.copy()
    config.update({
        'train_dir': args.train_dir,
        'test_dir': args.test_dir,
        'model_dir': args.model_dir,
        'result_dir': args.result_dir,
        'image_size': tuple(args.image_size),
        'num_components': args.num_components,
        'threshold': args.threshold,
        'augment': args.augment,
        'n_folds': args.n_folds,
    })

    print_banner()

    # 确保数据目录存在
    if not os.path.exists(config['train_dir']):
        print(f"[错误] 训练数据目录不存在: {config['train_dir']}")
        print("请确认数据集路径正确")
        sys.exit(1)

    # 根据参数选择运行模式
    if args.cross_validate:
        # 仅交叉验证
        run_cross_validation(config)

    elif args.train_only:
        # 仅训练
        model = run_train_pipeline(config)
        print("\n[OK] 模型训练完成!")

    elif args.test_only:
        # 仅测试 (需要已有模型)
        model = load_model(config['model_dir'])
        test_results = run_test_pipeline(model, config)
        print("\n[OK] 测试完成!")

    elif args.analyze_only:
        # 仅分析 (需要已有测试结果)
        results_path = os.path.join(config['result_dir'], 'test_results.pkl')
        if not os.path.exists(results_path):
            print(f"[错误] 测试结果文件不存在: {results_path}")
            print("请先运行 --test-only")
            sys.exit(1)

        with open(results_path, 'rb') as f:
            test_results = pickle.load(f)

        # 提取标签数据
        true_labels, pred_labels, confidences = [], [], []
        if 'functional' in test_results:
            for r in test_results['functional'].results:
                if r.test_name == 'Functional-Standard Test Set' and r.total_samples > 0:
                    true_labels = r.true_labels
                    pred_labels = r.pred_labels
                    confidences = r.confidences
                    break

        if not true_labels:
            print("[错误] 测试结果中没有找到有效的标签数据")
            sys.exit(1)

        info_path = os.path.join(config['model_dir'], 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
        else:
            model_info = {'num_classes': 'N/A', 'image_size': 'N/A',
                          'num_components': 'N/A', 'threshold': 'N/A'}

        analysis = run_analysis_pipeline(
            true_labels, pred_labels, confidences,
            test_results, model_info, config
        )
        print("\n[OK] 分析完成!")

    else:
        # 默认: 运行完整流程
        run_full_pipeline(config)


if __name__ == "__main__":
    main()
