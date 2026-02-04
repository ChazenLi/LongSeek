"""
Cross-Condition Generalization Framework - 跨条件泛化框架 (重构版)

参数分离策略:
- Mechanism Parameters (机制参数): {gamma, s, tau, alpha, beta}
  - 跨条件共享，代表生物机制
- Condition Parameters (条件参数): {D, lambda, S}
  - 每个条件独立，代表实验条件

测试协议:
1. spontaneous -> inhibitor
2. spontaneous -> opto
3. inhibitor -> opto

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
class ParameterSet:
    """参数集合"""
    # 机制参数 (跨条件共享)
    gamma: float = 0.5       # ERK耦合强度
    s: int = 1               # 符号开关
    tau: int = 1             # 时滞
    alpha: float = 1.0       # 迁移密度抑制
    beta: float = 1.0        # 分裂密度抑制

    # 条件参数 (每条件独立)
    p_move: float = 0.5      # 迁移概率
    p_div: float = 0.05      # 分裂概率
    edge_bonus: float = 2.0  # 边缘偏好
    D: float = 1.0           # 扩散系数
    lambda_: float = 1.0     # 分裂率系数
    S: float = 1.0           # 饱和密度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'gamma': self.gamma,
            's': self.s,
            'tau': self.tau,
            'alpha': self.alpha,
            'beta': self.beta,
            'p_move': self.p_move,
            'p_div': self.p_div,
            'edge_bonus': self.edge_bonus,
            'D': self.D,
            'lambda_': self.lambda_,
            'S': self.S,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ParameterSet':
        """从字典创建"""
        return cls(
            gamma=d.get('gamma', 0.5),
            s=int(d.get('s', 1)),
            tau=int(d.get('tau', 1)),
            alpha=d.get('alpha', 1.0),
            beta=d.get('beta', 1.0),
            p_move=d.get('p_move', 0.5),
            p_div=d.get('p_div', 0.05),
            edge_bonus=d.get('edge_bonus', 2.0),
            D=d.get('D', 1.0),
            lambda_=d.get('lambda_', 1.0),
            S=d.get('S', 1.0),
        )

    def get_mechanism_params(self) -> Dict[str, Any]:
        """获取机制参数"""
        return {
            'gamma': self.gamma,
            's': self.s,
            'tau': self.tau,
            'alpha': self.alpha,
            'beta': self.beta,
        }

    def get_condition_params(self) -> Dict[str, Any]:
        """获取条件参数"""
        return {
            'p_move': self.p_move,
            'p_div': self.p_div,
            'edge_bonus': self.edge_bonus,
            'D': self.D,
            'lambda_': self.lambda_,
            'S': self.S,
        }

    def update_mechanism(self, **kwargs):
        """更新机制参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_condition(self, **kwargs):
        """更新条件参数"""
        for k, v in kwargs.items():
            if k == 'lambda':
                k = 'lambda_'
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class CrossConditionResult:
    """跨条件泛化结果"""
    source_condition: str                     # 源条件
    target_condition: str                     # 目标条件
    mechanism_params: Dict[str, Any]          # 学到的机制参数
    source_loss: float                        # 源条件损失
    target_loss: float                        # 目标条件损失（泛化损失）
    transfer_loss: float                      # 损失增加
    success: bool                             # 是否成功泛化
    observables_source: Dict[str, np.ndarray]  # 源条件观测量
    observables_target: Dict[str, np.ndarray]  # 目标条件观测量


@dataclass
class CrossConditionSummary:
    """跨条件泛化汇总"""
    results: List[CrossConditionResult] = field(default_factory=list)
    mechanism_params_learned: Optional[Dict[str, Any]] = None
    best_transfer: Optional[str] = None
    worst_transfer: Optional[str] = None
    overall_success_rate: float = 0.0


class CrossConditionFramework:
    """
    跨条件泛化框架

    测试机制参数的可迁移性
    """

    # 条件配置
    CONDITION_CONFIGS = {
        'spontaneous': {
            'name': 'Figure4A Spontaneous ERK Wave',
            'dataset_key': 'figure4a_erk',
            'expected_behavior': '正常ERK波，s=+1'
        },
        'inhibitor': {
            'name': 'FigureS1A ADAM10 Inhibitor',
            'dataset_key': 'figureS1a_inhibitor',
            'expected_behavior': 'ERK波减弱，gamma降低'
        },
        'opto': {
            'name': 'FigureS4 Optogenetic',
            'dataset_key': 'figureS4_ratio',
            'expected_behavior': 'ERK波增强，gamma升高'
        },
    }

    def __init__(self,
                 datasets: Dict[str, Any],
                 loss_fn: Optional[Any] = None,
                 ca_runner: Optional[Any] = None):
        """
        Args:
            datasets: 数据集字典 {condition: ERKDataset}
            loss_fn: 损失函数
            ca_runner: CA运行器
        """
        self.datasets = datasets
        self.loss_fn = loss_fn
        self.ca_runner = ca_runner

    def test_transfer(self,
                     source_condition: str,
                     target_condition: str,
                     params: ParameterSet) -> CrossConditionResult:
        """
        测试从源条件到目标条件的泛化

        Args:
            source_condition: 源条件名称
            target_condition: 目标条件名称
            params: 参数集合

        Returns:
            CrossConditionResult
        """
        logger.info(f"Testing transfer: {source_condition} -> {target_condition}")

        # 获取数据
        source_data = self.datasets.get(source_condition)
        target_data = self.datasets.get(target_condition)

        if source_data is None or target_data is None:
            raise ValueError(f"Missing data for {source_condition} or {target_condition}")

        # 1. 在源条件上训练/评估
        source_loss = self._evaluate_condition(
            source_condition, params
        )

        # 2. 在目标条件上测试（使用相同的机制参数）
        # 只调整条件参数
        target_params = ParameterSet(**params.get_mechanism_params())
        # 使用目标条件的数据来估计合适的条件参数
        target_params = self._estimate_condition_params(target_data, target_params)

        target_loss = self._evaluate_condition(
            target_condition, target_params
        )

        # 计算转移损失
        transfer_loss = target_loss - source_loss
        success = transfer_loss < 0.5  # 阈值

        result = CrossConditionResult(
            source_condition=source_condition,
            target_condition=target_condition,
            mechanism_params=params.get_mechanism_params(),
            source_loss=source_loss,
            target_loss=target_loss,
            transfer_loss=transfer_loss,
            success=success,
            observables_source={},
            observables_target={}
        )

        logger.info(f"  Source loss: {source_loss:.6f}")
        logger.info(f"  Target loss: {target_loss:.6f}")
        logger.info(f"  Transfer loss: {transfer_loss:+.6f}")
        logger.info(f"  Success: {success}")

        return result

    def run_all_transfers(self,
                         initial_params: Optional[ParameterSet] = None
                         ) -> CrossConditionSummary:
        """
        运行所有跨条件转移测试

        测试协议：
        1. spontaneous -> inhibitor
        2. spontaneous -> opto
        3. inhibitor -> opto

        Returns:
            CrossConditionSummary
        """
        if initial_params is None:
            initial_params = ParameterSet()

        summary = CrossConditionSummary()
        results = []

        # 测试协议
        test_protocols = [
            ('spontaneous', 'inhibitor'),
            ('spontaneous', 'opto'),
            ('inhibitor', 'opto'),
        ]

        for source, target in test_protocols:
            if source in self.datasets and target in self.datasets:
                result = self.test_transfer(source, target, initial_params)
                results.append(result)

        summary.results = results

        # 分析结果
        if results:
            # 最好的转移（最小转移损失）
            best = min(results, key=lambda r: r.transfer_loss)
            summary.best_transfer = f"{best.source_condition} -> {best.target_condition}"

            # 最差的转移
            worst = max(results, key=lambda r: r.transfer_loss)
            summary.worst_transfer = f"{worst.source_condition} -> {worst.target_condition}"

            # 成功率
            success_count = sum(1 for r in results if r.success)
            summary.overall_success_rate = success_count / len(results)

            # 学到的机制参数（使用spontaneous的结果）
            spontaneous_result = next(
                (r for r in results if r.source_condition == 'spontaneous'),
                None
            )
            if spontaneous_result:
                summary.mechanism_params_learned = spontaneous_result.mechanism_params

        return summary

    def _evaluate_condition(self,
                           condition: str,
                           params: ParameterSet) -> float:
        """在指定条件下评估参数"""
        data = self.datasets[condition]

        # 运行CA模拟
        if self.ca_runner is None:
            from core.ca.phase2_teacher_forcing import PhaseIITeacherForcingCA, CAParamsExtended, CouplingParams

            coupling = CouplingParams(
                gamma=params.gamma,
                s=params.s,
                tau=params.tau
            )

            ca_params = CAParamsExtended(
                p_move=params.p_move,
                p_div=params.p_div,
                alpha=params.alpha,
                beta=params.beta,
                edge_bonus=params.edge_bonus,
                coupling=coupling
            )

            H, W = data.mask.shape
            ca = PhaseIITeacherForcingCA(H, W, ca_params, data.erk_field)
            ca.initialize_from_mask(data.mask)

            num_steps = len(data.time_points_hours)
            history = ca.run(num_steps, erk_field_sequence=data.erk_field)
        else:
            history = self.ca_runner(params.to_dict(), data.mask, data.erk_field, len(data.time_points_hours))

        # 计算损失
        if self.loss_fn is None:
            from core.inference.erk_objective import multi_objective_loss
            loss_result = multi_objective_loss(history, data.observables, data.time_points_hours, data.erk_field)
            return loss_result.total
        else:
            loss, _ = self.loss_fn(history, data.observables, data.time_points_hours)
            return loss

    def _estimate_condition_params(self,
                                  data: Any,
                                  params: ParameterSet) -> ParameterSet:
        """从数据估计条件参数"""
        # 简化版本：基于条件类型调整
        # 实际应用中可以从数据拟合

        condition_type = data.metadata.condition

        if condition_type == 'inhibitor':
            # 抑制剂条件：ERK波减弱
            params.p_move *= 0.8  # 降低迁移率
            params.p_div *= 0.9   # 略微降低分裂率
        elif condition_type == 'optogenetic':
            # 光遗传条件：ERK波增强
            params.p_move *= 1.2  # 提高迁移率
            params.p_div *= 1.1   # 略微提高分裂率

        return params

    def print_summary(self, summary: CrossConditionSummary):
        """打印跨条件泛化摘要"""
        print("\n" + "=" * 60)
        print("CROSS-CONDITION GENERALIZATION SUMMARY")
        print("=" * 60)

        if summary.mechanism_params_learned:
            print("\nLearned Mechanism Parameters:")
            for k, v in summary.mechanism_params_learned.items():
                print(f"  {k}: {v}")

        print("\nTransfer Results:")
        print("-" * 50)
        print(f"{'Source -> Target':<25} {'Transfer Loss':>15} {'Success':>10}")
        print("-" * 50)

        for result in summary.results:
            status = "✓" if result.success else "✗"
            print(f"{result.source_condition} -> {result.target_condition:<15} {result.transfer_loss:>+15.6f} {status:>10}")

        print(f"\nOverall Success Rate: {summary.overall_success_rate:.1%}")
        print(f"Best Transfer: {summary.best_transfer}")
        print(f"Worst Transfer: {summary.worst_transfer}")

        print("\n" + "=" * 60)


def fit_cross_condition_params(datasets: Dict[str, Any],
                                source_condition: str = 'spontaneous') -> ParameterSet:
    """
    在源条件上拟合参数

    Args:
        datasets: 数据集字典
        source_condition: 源条件名称

    Returns:
        拟合得到的参数
    """
    # 简化版本：返回默认参数
    # 实际应用中应该使用优化算法拟合

    params = ParameterSet()

    # 根据条件类型调整
    if source_condition == 'spontaneous':
        params.gamma = 0.5
        params.s = 1
        params.tau = 1
    elif source_condition == 'inhibitor':
        params.gamma = 0.3  # ERK波减弱
        params.s = 1
        params.tau = 1
    elif source_condition == 'optogenetic':
        params.gamma = 0.8  # ERK波增强
        params.s = 1
        params.tau = 0

    return params


if __name__ == "__main__":
    # 测试跨条件泛化
    print("Testing Cross-Condition Generalization Framework")

    # 创建模拟数据
    from core.preprocess.erk_dataset_loader import ERKDataset, ERKDatasetMetadata

    H, W, T = 50, 50, 60

    # spontaneous数据
    erk_spont = np.random.rand(T, H, W).astype(np.float32)
    mask_spont = np.ones((H, W), dtype=np.float32)
    mask_spont[20:30, 20:30] = 0

    dataset_spont = ERKDataset(
        erk_field=erk_spont,
        mask=mask_spont,
        metadata=ERKDatasetMetadata(
            dataset_key='spontaneous',
            condition='spontaneous',
            dt_seconds=180.0,
            pixel_size_um=0.49,
            num_frames=T,
            shape=(T, H, W)
        ),
        time_points_hours=np.arange(T) * (180.0 / 3600.0),
        observables={
            'A_t': np.linspace(100, 50, T),
            'roughness': np.ones(T) * 0.5,
        }
    )

    # inhibitor数据
    erk_inhib = erk_spont * 0.7  # ERK波减弱
    dataset_inhib = ERKDataset(
        erk_field=erk_inhib,
        mask=mask_spont.copy(),
        metadata=ERKDatasetMetadata(
            dataset_key='inhibitor',
            condition='inhibitor',
            dt_seconds=120.0,
            pixel_size_um=0.49,
            num_frames=T,
            shape=(T, H, W)
        ),
        time_points_hours=np.arange(T) * (120.0 / 3600.0),
        observables={
            'A_t': np.linspace(100, 60, T),
            'roughness': np.ones(T) * 0.6,
        }
    )

    datasets = {
        'spontaneous': dataset_spont,
        'inhibitor': dataset_inhib,
    }

    # 创建框架
    framework = CrossConditionFramework(datasets)

    # 运行跨条件测试
    summary = framework.run_all_transfers()

    # 打印摘要
    framework.print_summary(summary)

    print("\n✅ Cross-Condition Generalization test complete!")
