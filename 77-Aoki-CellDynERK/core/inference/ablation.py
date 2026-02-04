"""
Ablation Study Framework - 消融实验框架 (重构版)

消融实验用于验证每个组件的重要性：
- no_coupling: γ=0 (无ERK耦合)
- no_lag: τ=0 (无时滞)
- s_fixed_plus1: s=+1固定
- s_fixed_minus1: s=-1固定
- no_erk: 使用Phase I (无ERK场)
- no_density: α=0, β=0 (无密度抑制)
- no_edge: edge_bonus=0 (无边缘偏好)

作者: IF-CA Team
日期: 2026-02-04
版本: 2.0 (Refactored)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str                          # 配置名称
    params: Dict[str, Any]             # 参数修改
    description: str                    # 描述


# 标准消融配置
ABLATION_CONFIGS = {
    'baseline': AblationConfig(
        name='baseline',
        params={},
        description='完整模型（所有机制启用）'
    ),
    'no_coupling': AblationConfig(
        name='no_coupling',
        params={'gamma': 0.0},
        description='无ERK耦合 (γ=0)'
    ),
    'no_lag': AblationConfig(
        name='no_lag',
        params={'tau': 0},
        description='无时滞 (τ=0)'
    ),
    's_fixed_plus1': AblationConfig(
        name='s_fixed_plus1',
        params={'s': +1, 's_trainable': False},
        description='s固定为+1（沿梯度迁移）'
    ),
    's_fixed_minus1': AblationConfig(
        name='s_fixed_minus1',
        params={'s': -1, 's_trainable': False},
        description='s固定为-1（逆梯度迁移）'
    ),
    'no_erk': AblationConfig(
        name='no_erk',
        params={'use_erk': False},
        description='无ERK场 (Phase I模型)'
    ),
    'no_density': AblationConfig(
        name='no_density',
        params={'alpha': 0.0, 'beta': 0.0},
        description='无密度抑制'
    ),
    'no_edge': AblationConfig(
        name='no_edge',
        params={'edge_bonus': 1.0},
        description='无边缘偏好'
    ),
}


@dataclass
class AblationResult:
    """单个消融实验结果"""
    config_name: str                           # 配置名称
    params: Dict[str, Any]                     # 使用的参数
    loss_total: float                          # 总损失
    loss_components: Dict[str, float]          # 各分量损失
    delta_from_baseline: float                 # 相对于baseline的变化
    observables: Dict[str, np.ndarray]         # 模拟得到的观测量


@dataclass
class AblationStudySummary:
    """消融实验汇总"""
    baseline_result: Optional[AblationResult] = None
    ablation_results: Dict[str, AblationResult] = field(default_factory=dict)
    ranking: List[Tuple[str, float]] = field(default_factory=list)  # (config_name, delta)
    key_findings: List[str] = field(default_factory=list)


def run_ablation_study(
    baseline_params: Dict[str, Any],
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    erk_field: np.ndarray,
    ablation_configs: Optional[Dict[str, AblationConfig]] = None,
    ca_runner: Optional[Any] = None,
    loss_fn: Optional[Any] = None
) -> AblationStudySummary:
    """
    运行消融实验

    Args:
        baseline_params: 基线参数
        initial_mask: 初始mask
        observed: 观测数据
        observed_time: 观测时间点
        erk_field: ERK场
        ablation_configs: 消融配置（默认使用标准配置）
        ca_runner: CA运行器
        loss_fn: 损失函数

    Returns:
        AblationStudySummary
    """
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS

    summary = AblationStudySummary()

    # 1. 运行baseline
    logger.info("Running baseline simulation...")
    baseline_result = _run_single_ablation(
        'baseline',
        baseline_params,
        initial_mask,
        observed,
        observed_time,
        erk_field,
        ca_runner,
        loss_fn
    )
    summary.baseline_result = baseline_result
    summary.ablation_results['baseline'] = baseline_result

    baseline_loss = baseline_result.loss_total
    logger.info(f"Baseline loss: {baseline_loss:.6f}")

    # 2. 运行各消融配置
    deltas = []

    for config_name, config in ablation_configs.items():
        if config_name == 'baseline':
            continue

        logger.info(f"Running ablation: {config_name} - {config.description}")

        # 合并参数
        ablation_params = {**baseline_params, **config.params}

        # 运行
        result = _run_single_ablation(
            config_name,
            ablation_params,
            initial_mask,
            observed,
            observed_time,
            erk_field,
            ca_runner,
            loss_fn
        )

        # 计算delta
        delta = result.loss_total - baseline_loss
        result.delta_from_baseline = delta

        summary.ablation_results[config_name] = result
        deltas.append((config_name, delta))

        logger.info(f"  Loss: {result.loss_total:.6f}, Delta: {delta:+.6f}")

    # 3. 排序（按delta从小到大）
    summary.ranking = sorted(deltas, key=lambda x: x[1])

    # 4. 生成关键发现
    summary.key_findings = _generate_key_findings(summary)

    return summary


def _run_single_ablation(
    config_name: str,
    params: Dict[str, Any],
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    erk_field: np.ndarray,
    ca_runner: Any,
    loss_fn: Any
) -> AblationResult:
    """运行单个消融实验"""
    # 如果没有提供CA runner，使用默认的
    if ca_runner is None:
        from core.ca.phase2_teacher_forcing import PhaseIITeacherForcingCA, CAParamsExtended, CouplingParams

        # 检查是否使用ERK
        use_erk = params.get('use_erk', True)

        if use_erk:
            coupling = CouplingParams(
                gamma=params.get('gamma', 0.5),
                s=params.get('s', 1),
                tau=params.get('tau', 1)
            )
        else:
            coupling = CouplingParams(gamma=0.0, s=1, tau=0)

        ca_params = CAParamsExtended(
            p_move=params.get('p_move', 0.5),
            p_div=params.get('p_div', 0.05),
            alpha=params.get('alpha', 1.0),
            beta=params.get('beta', 1.0),
            edge_bonus=params.get('edge_bonus', 2.0),
            coupling=coupling
        )

        H, W = initial_mask.shape
        ca = PhaseIITeacherForcingCA(H, W, ca_params, erk_field if use_erk else None)
        ca.initialize_from_mask(initial_mask)

        # 运行
        num_steps = len(observed_time)
        erk_sequence = erk_field if use_erk else None
        history = ca.run(num_steps, erk_field_sequence=erk_sequence)
    else:
        # 使用提供的runner
        history = ca_runner(params, initial_mask, erk_field, len(observed_time))

    # 计算损失
    if loss_fn is None:
        from core.inference.erk_objective import multi_objective_loss, LossWeights
        loss_result = multi_objective_loss(history, observed, observed_time, erk_field)
        loss_total = loss_result.total
        loss_components = loss_result.to_dict()
    else:
        loss_total, loss_components = loss_fn(history, observed, observed_time)

    return AblationResult(
        config_name=config_name,
        params=params,
        loss_total=loss_total,
        loss_components=loss_components,
        delta_from_baseline=0.0,  # 将在后面设置
        observables={}
    )


def _generate_key_findings(summary: AblationStudySummary) -> List[str]:
    """生成关键发现"""
    findings = []

    if not summary.ranking:
        return findings

    # 最重要的发现：delta最大的消融
    worst_ablation = summary.ranking[-1]
    findings.append(f"Most important mechanism: {worst_ablation[0]} (Δ={worst_ablation[1]:.4f})")

    # 检查ERK耦合的重要性
    if 'no_coupling' in summary.ablation_results:
        delta = summary.ablation_results['no_coupling'].delta_from_baseline
        if delta > 0:
            findings.append(f"ERK coupling is critical: removing it increases loss by {delta:.4f}")
        else:
            findings.append(f"ERK coupling has minimal effect (Δ={delta:.4f})")

    # 检查时滞的重要性
    if 'no_lag' in summary.ablation_results:
        delta = summary.ablation_results['no_lag'].delta_from_baseline
        findings.append(f"Time lag effect: Δ={delta:.4f}")

    # 检查s参数
    if 's_fixed_plus1' in summary.ablation_results and 's_fixed_minus1' in summary.ablation_results:
        delta_plus = summary.ablation_results['s_fixed_plus1'].delta_from_baseline
        delta_minus = summary.ablation_results['s_fixed_minus1'].delta_from_baseline

        if delta_plus < delta_minus:
            findings.append(f"s=+1 is more consistent with data than s=-1 (Δ: {delta_plus:.4f} vs {delta_minus:.4f})")
        elif delta_minus < delta_plus:
            findings.append(f"s=-1 is more consistent with data than s=+1 (Δ: {delta_minus:.4f} vs {delta_plus:.4f})")

    return findings


def print_ablation_summary(summary: AblationStudySummary):
    """打印消融实验摘要"""
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    if summary.baseline_result:
        print(f"\nBaseline Loss: {summary.baseline_result.loss_total:.6f}")

    print("\nAblation Results (sorted by impact):")
    print("-" * 40)
    print(f"{'Config':<20} {'Delta':>12} {'Total Loss':>12}")
    print("-" * 40)

    for config_name, delta in summary.ranking:
        result = summary.ablation_results[config_name]
        print(f"{config_name:<20} {delta:>+12.6f} {result.loss_total:>12.6f}")

    print("\nKey Findings:")
    for i, finding in enumerate(summary.key_findings, 1):
        print(f"  {i}. {finding}")

    print("\n" + "=" * 60)


def get_component_importance(summary: AblationStudySummary) -> Dict[str, float]:
    """
    获取各组件的重要性

    Returns:
        {component: importance_score}
        importance_score = loss_without_component - loss_baseline
    """
    importance = {}

    for config_name, result in summary.ablation_results.items():
        if config_name == 'baseline':
            continue
        importance[config_name] = result.delta_from_baseline

    # 按重要性排序
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# 便捷函数
def quick_ablation(params: Dict[str, Any],
                   initial_mask: np.ndarray,
                   observed: Dict[str, np.ndarray],
                   observed_time: np.ndarray,
                   erk_field: np.ndarray) -> AblationStudySummary:
    """
    快速消融实验（使用默认配置）

    只测试最重要的几个消融：
    - baseline
    - no_coupling
    - no_lag
    - s_fixed_plus1
    """
    quick_configs = {
        'baseline': ABLATION_CONFIGS['baseline'],
        'no_coupling': ABLATION_CONFIGS['no_coupling'],
        'no_lag': ABLATION_CONFIGS['no_lag'],
        's_fixed_plus1': ABLATION_CONFIGS['s_fixed_plus1'],
    }

    return run_ablation_study(
        params, initial_mask, observed, observed_time, erk_field,
        ablation_configs=quick_configs
    )


if __name__ == "__main__":
    # 测试消融实验
    print("Testing Ablation Study Framework")

    # 创建模拟数据
    H, W = 50, 50
    T = 60

    initial_mask = np.ones((H, W), dtype=np.float32)
    initial_mask[20:30, 20:30] = 0

    erk_field = np.random.rand(T, H, W).astype(np.float32)

    observed_time = np.linspace(0, 3, T)
    observed = {
        'A_t': np.linspace(100, 50, T),
        'roughness': np.ones(T) * 0.5,
        'angle_v_grad_C': np.ones(T) * 0.3,
    }

    baseline_params = {
        'p_move': 0.5,
        'p_div': 0.05,
        'alpha': 1.0,
        'beta': 1.0,
        'edge_bonus': 2.0,
        'gamma': 0.5,
        's': 1,
        'tau': 1,
    }

    # 运行快速消融
    summary = quick_ablation(
        baseline_params, initial_mask, observed, observed_time, erk_field
    )

    # 打印摘要
    print_ablation_summary(summary)

    print("\n✅ Ablation Study test complete!")
