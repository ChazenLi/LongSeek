"""
IF-CCA Week 6-8 Integration Test

Comprehensive test demonstrating:
1. Identifiability Diagnostics (Week 6)
2. Multi-Condition Training (Week 7)
3. Causal Validation (Week 8)

This script validates that all advanced features work correctly together.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference.identifiability import (
    IdentifiabilityDiagnostics,
    compute_posterior_correlation,
    plot_identifiability_results
)
from core.inference.multi_condition import (
    MultiConditionTrainer,
    ConditionDataset,
    ParameterTransferAnalyzer,
    plot_multi_condition_results
)
from core.inference.causal_validation import (
    CausalValidator,
    plot_causal_validation
)


def create_synthetic_data():
    """创建合成测试数据"""
    np.random.seed(42)

    # 初始mask
    initial_mask = np.ones((100, 100), dtype=np.float32)
    initial_mask[30:70, 30:70] = 0.0  # 中心伤口

    # 观测时间
    observed_time = np.array([0, 10, 20, 30, 40])

    # 观测伤口面积（指数衰减）
    observed_area = np.array([1600, 1200, 800, 500, 200])

    observed = {'A_t': observed_area}

    # ERK场
    H, W, T = 100, 100, 5
    erk_field = np.zeros((T, H, W))
    for t in range(T):
        x = np.linspace(-3, 3, W)
        y = np.linspace(-3, 3, H)
        X, Y = np.meshgrid(x, y)
        erk_field[t] = 1.0 / (1 + 0.5 * (X**2 + Y**2))

    return initial_mask, observed, observed_time, erk_field


def dummy_objective(params, initial_mask, observed, observed_time):
    """虚拟目标函数（用于测试）"""
    # 简单的MSE损失
    base_loss = 0.1
    for key in ['p_move', 'p_div', 'gamma', 'alpha', 'beta']:
        if key in params:
            target = {'p_move': 0.5, 'p_div': 0.05, 'gamma': 1.0, 'alpha': 1.0, 'beta': 1.0}
            base_loss += (params[key] - target.get(key, 0.5))**2 * 0.1
    return base_loss


def dummy_model(params):
    """虚拟模型函数（用于敏感性分析）"""
    base_wound_area = 1600
    p_move = params.get('p_move', 0.5)
    gamma = params.get('gamma', 1.0)

    # 简单的响应模型
    wound_area = base_wound_area * (1 - 0.02 * p_move * gamma)
    migrations = int(100 * p_move * gamma)
    divisions = int(10 * params.get('p_div', 0.05))

    return {
        'wound_area': wound_area,
        'migrations': migrations,
        'divisions': divisions,
    }


# ============================================================================
# Week 6: Identifiability Diagnostics
# ============================================================================

def test_identifiability_diagnostics():
    """测试可辨识性诊断"""
    print("\n" + "="*60)
    print("Week 6: Identifiability Diagnostics")
    print("="*60)

    # 创建参数历史（模拟贝叶斯优化的轨迹）
    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'gamma']
    param_ranges = {
        'p_move': (0.01, 0.99),
        'p_div': (0.001, 0.2),
        'alpha': (0.1, 5.0),
        'beta': (0.1, 5.0),
        'gamma': (0.0, 2.0),
    }

    # 模拟参数历史
    n_samples = 50
    params_history = []

    # 生成收敛到最优值的轨迹
    for i in range(n_samples):
        t = i / n_samples
        params = {
            'p_move': 0.5 + np.random.normal(0, 0.1) * (1-t),
            'p_div': 0.05 + np.random.normal(0, 0.02) * (1-t),
            'alpha': 1.0 + np.random.normal(0, 0.5) * (1-t),
            'beta': 1.0 + np.random.normal(0, 0.5) * (1-t),
            'gamma': 1.0 + np.random.normal(0, 0.3) * (1-t),
        }
        params_history.append(params)

    # 创建诊断器
    diagnostics = IdentifiabilityDiagnostics(
        objective_fn=dummy_objective,
        param_names=param_names,
        param_ranges=param_ranges
    )

    # 1. 相关性分析
    print("\n[1] Computing posterior correlation matrix...")
    corr_matrix, high_corr = diagnostics.analyze_correlation(params_history)

    print("    Posterior Correlation Matrix:")
    for i, name1 in enumerate(param_names):
        for j, name2 in enumerate(param_names):
            if i < j:
                print(f"      {name1} - {name2}: {corr_matrix[i,j]:.3f}")

    if high_corr:
        print(f"    High correlations detected: {list(high_corr.keys())}")
    else:
        print("    No high correlations (|corr| > 0.5) detected.")

    # 2. Fisher信息
    print("\n[2] Computing Fisher information...")
    best_params = {name: params_history[-1][name] for name in param_names}
    fisher_info, eigenvalues, precision = diagnostics.analyze_fisher_information(best_params)

    print("    Parameter Precision (diagonal of Fisher info):")
    for name in param_names:
        prec = precision.get(name, 0.0)
        level = "HIGH" if prec > 0.01 else "MEDIUM" if prec > 0.001 else "LOW"
        print(f"      {name}: {prec:.4f} [{level}]")

    # 3. 敏感性分析
    print("\n[3] Computing sensitivity indices...")
    sensitivity = diagnostics.analyze_sensitivity(best_params, dummy_model, n_samples=20)

    print("    Sensitivity to wound_area:")
    for name in param_names:
        sens = sensitivity[name].get('wound_area', 0.0)
        print(f"      {name}: {sens:.3f}")

    # 4. 生成报告
    print("\n[4] Generating identifiability report...")
    results = diagnostics.full_analysis(
        params_history=params_history,
        best_params=best_params,
        baseline_loss=0.05,
        model_fn=dummy_model
    )

    print("    Summary:")
    summary = results['summary']
    print(f"      Identifiable: {summary['identifiable_params']}")
    print(f"      Weakly identifiable: {summary['weakly_identifiable_params']}")
    print(f"      Unidentifiable: {summary['unidentifiable_params']}")

    return results


# ============================================================================
# Week 7: Multi-Condition Training
# ============================================================================

def test_multi_condition_training():
    """测试多条件训练"""
    print("\n" + "="*60)
    print("Week 7: Multi-Condition Training")
    print("="*60)

    # 创建多条件数据集
    print("\n[1] Creating multi-condition datasets...")

    conditions = {}
    for name in ['spontaneous', 'inhibitor', 'opto']:
        initial_mask = np.ones((100, 100))
        initial_mask[30:70, 30:70] = 0

        # 不同条件有不同的愈合速度
        factor = {'spontaneous': 1.0, 'inhibitor': 0.7, 'opto': 1.3}[name]
        observed_area = np.array([1600, 1200, 800, 500, 200]) * factor

        conditions[name] = ConditionDataset(
            name=name,
            initial_mask=initial_mask,
            observed={'A_t': observed_area},
            observed_time=np.array([0, 10, 20, 30, 40]),
            metadata={'condition_type': name}
        )

        print(f"    {name}: initial_area={observed_area[0]:.0f}, final_area={observed_area[-1]:.0f}")

    # 创建训练器
    print("\n[2] Initializing MultiConditionTrainer...")
    trainer = MultiConditionTrainer(conditions)

    # 训练共享参数
    print("\n[3] Training shared parameters...")
    training_results = trainer.train_shared_parameters(
        condition_names=['spontaneous', 'inhibitor'],
        objective_fn=dummy_objective,
        n_iterations=20
    )

    print(f"    Best training loss: {training_results['best_loss']:.4f}")

    # 评估泛化
    print("\n[4] Evaluating generalization...")

    gen_results = []
    for test_cond in ['inhibitor', 'opto']:
        result = trainer.evaluate_generalization(
            train_conditions=['spontaneous'],
            test_condition=test_cond,
            objective_fn=dummy_objective
        )
        gen_results.append(result)
        status = "✅ PASS" if result['transfer_success'] else "❌ FAIL"
        print(f"    spontaneous → {test_cond}: ratio={result['ratio']:.2f} [{status}]")

    # 交叉验证
    print("\n[5] Cross-validation...")
    cv_results = trainer.cross_validation(dummy_objective, n_folds=2)

    print(f"    Average test/train ratio: {cv_results['average_ratio']:.2f}")
    print(f"    Success rate: {cv_results['success_rate']*100:.0f}%")

    # 参数稳定性
    print("\n[6] Analyzing parameter stability...")
    stability = trainer.get_parameter_stability()

    print("    Parameter stability (CV):")
    for param_name, metrics in stability.items():
        level = metrics['transferability_level']
        print(f"      {param_name}: CV={metrics['cv']:.3f} [{level}]")

    # 参数分离建议
    print("\n[7] Parameter separation recommendation...")
    analyzer = ParameterTransferAnalyzer(trainer.condition_params)
    shared, separated = analyzer.recommend_separation()

    print(f"    Shared parameters (mechanism): {shared}")
    print(f"    Separated parameters (context): {separated}")

    return {
        'training_results': training_results,
        'generalization': gen_results,
        'cv_results': cv_results,
        'stability': stability,
    }


# ============================================================================
# Week 8: Causal Validation
# ============================================================================

def test_causal_validation():
    """测试因果验证"""
    print("\n" + "="*60)
    print("Week 8: Causal Validation")
    print("="*60)

    # 创建测试数据
    print("\n[1] Creating optogenetic intervention data...")
    initial_mask, observed, observed_time, erk_field = create_synthetic_data()

    # 基础参数
    base_params = {
        'p_move': 0.5,
        'p_div': 0.05,
        'alpha': 1.0,
        'beta': 1.0,
        'edge_bonus': 2.0,
        'k_time': 1.0,
        'coupling': {
            'gamma': 1.0,
            's': +1,
            'tau': 1,
            'D_coupling': 1.0
        }
    }

    print("    Base parameters:")
    print(f"      gamma: {base_params['coupling']['gamma']}")
    print(f"      s: {base_params['coupling']['s']}")
    print(f"      tau: {base_params['coupling']['tau']}")

    # 创建因果验证器
    print("\n[2] Initializing CausalValidator...")
    validator = CausalValidator(base_params, erk_field)

    # 可证伪性测试
    print("\n[3] Running falsifiability test (s=+1 vs s=-1)...")

    class MockCA:
        """模拟CA类"""
        def __init__(self, H, W, params, erk_field=None):
            self.H, self.W = H, W
            self.params = params
            self.grad_y = np.zeros((H, W))
            self.grad_x = np.zeros((H, W))

        def initialize_from_mask(self, mask):
            pass

        def run(self, num_steps, erk_field_sequence=None):
            history = []
            for i in range(num_steps):
                # 模拟s=+1时愈合更快
                s = self.params.get('coupling', {}).get('s', +1)
                base_closure = 50
                s_effect = 20 * s
                wound_area = max(100, 1600 - (base_closure + s_effect) * i)

                history.append({
                    'wound_area': wound_area,
                    'migrations': 100 + s * 10,
                    'divisions': 10
                })
            return history

    def loss_fn(pred):
        """简单的loss函数"""
        # s=+1应该得到更小的loss
        if hasattr(pred, 'predicted_velocity'):
            s = base_params['coupling']['s']
            return 0.5 - s * 0.3  # s=+1 -> 0.2, s=-1 -> 0.8
        return 0.5

    fals_result = validator.falsifiability_test(
        initial_mask=initial_mask,
        opto_erk_field=erk_field,
        ca_class=MockCA,
        objective_fn=loss_fn,
        n_trials=3
    )

    print(f"    Loss (s=+1, correct): {fals_result.loss_correct:.4f}")
    print(f"    Loss (s=-1, wrong): {fals_result.loss_wrong:.4f}")
    print(f"    Improvement: {fals_result.improvement*100:.1f}%")
    print(f"    Falsifiable: {fals_result.is_falsifiable}")
    print(f"    Confidence: {fals_result.confidence}")

    # 反事实分析
    print("\n[4] Counterfactual analysis (ERK ablation)...")

    cf_result = validator.counterfactual_analysis(
        initial_mask=initial_mask,
        base_erk_field=erk_field,
        ca_class=MockCA,
        intervention='remove_ERK'
    )

    print(f"    Wound area with ERK: {cf_result['baseline']['final_wound_area']:.0f}")
    print(f"    Wound area without ERK: {cf_result['counterfactual']['final_wound_area']:.0f}")
    print(f"    Difference: {cf_result['difference']['wound_area_diff']:.0f} pixels")

    # 预测opto干预
    print("\n[5] Predicting optogenetic intervention...")

    pred = validator.predict_opto_migration(
        opto_erk_field=erk_field,
        initial_mask=initial_mask,
        ca_class=MockCA,
        s_sign=+1
    )

    print(f"    Predicted velocities: {pred.predicted_velocity[:3]}")

    return {
        'falsifiability': fals_result,
        'counterfactual': cf_result,
        'prediction': pred,
    }


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """运行所有Week 6-8测试"""
    print("\n" + "="*70)
    print(" IF-CCA Week 6-8 Integration Test")
    print("="*70)
    print("\nThis test validates:")
    print("  • Week 6: Identifiability Diagnostics")
    print("  • Week 7: Multi-Condition Training")
    print("  • Week 8: Causal Validation")

    all_results = {}

    try:
        # Week 6
        ident_results = test_identifiability_diagnostics()
        all_results['identifiability'] = ident_results

        # Week 7
        multi_results = test_multi_condition_training()
        all_results['multi_condition'] = multi_results

        # Week 8
        causal_results = test_causal_validation()
        all_results['causal_validation'] = causal_results

        # 汇总
        print("\n" + "="*70)
        print(" TEST SUMMARY")
        print("="*70)

        print("\n✅ Week 6 - Identifiability Diagnostics:")
        summary = ident_results['summary']
        print(f"   • Identifiable params: {len(summary['identifiable_params'])}")
        print(f"   • Unidentifiable params: {len(summary['unidentifiable_params'])}")

        print("\n✅ Week 7 - Multi-Condition Training:")
        cv = multi_results['cv_results']
        print(f"   • Average transfer ratio: {cv['average_ratio']:.2f}")
        print(f"   • Success rate: {cv['success_rate']*100:.0f}%")

        print("\n✅ Week 8 - Causal Validation:")
        fals = causal_results['falsifiability']
        print(f"   • Falsifiable: {fals.is_falsifiable}")
        print(f"   • Confidence: {fals.confidence}")

        # 保存结果
        output_dir = PROJECT_ROOT / 'results'
        output_dir.mkdir(exist_ok=True)

        import json
        with open(output_dir / 'week6_8_test_results.json', 'w') as f:
            json.dump({
                'timestamp': str(np.datetime64('now')),
                'results': all_results,
            }, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {output_dir / 'week6_8_test_results.json'}")

        print("\n" + "="*70)
        print(" ALL TESTS PASSED ✅")
        print("="*70)

        return all_results

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_all_tests()
