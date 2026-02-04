"""
Multi-Condition Training Framework (Week 7)

Implements joint training across multiple experimental conditions:
- Spontaneous (baseline)
- Inhibitor (drug perturbation)
- Optogenetic (controlled intervention)

Key Idea:
- Shared mechanism parameters: {gamma, s, tau} (identifiable)
- Condition-specific parameters: {p_move, k_time, edge_bonus}

Hypothesis:
If mechanism parameters are truly identifiable, they should:
1. Transfer across conditions (L_test/L_train < 1.2)
2. Show consistent values when trained on different conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ConditionDataset:
    """单个条件的数据集"""
    name: str
    initial_mask: np.ndarray
    observed: Dict[str, np.ndarray]
    observed_time: np.ndarray
    erk_field: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class MultiConditionConfig:
    """多条件训练配置"""
    # 共享机制参数（可辨识）
    shared_gamma: float = 1.0
    shared_s: int = +1
    shared_tau: int = 1
    shared_alpha: float = 1.0
    shared_beta: float = 1.0

    # 条件特异参数
    p_move_base: float = 0.5
    p_div_base: float = 0.05
    edge_bonus_base: float = 2.0
    k_time_base: float = 1.0

    # 允许的条件间变异
    p_move_var: float = 0.1
    k_time_var: float = 0.2
    edge_bonus_var: float = 0.5


class MultiConditionTrainer:
    """
    多条件联合训练框架

    核心思想：
    - 不同实验条件（spontaneous, inhibitor, opto）可能共享相同的分子机制
    - 但细胞响应能力（迁移概率、时间尺度等）可能不同
    - 通过联合训练，可以区分机制参数和条件参数
    """

    def __init__(self,
                 conditions: Dict[str, ConditionDataset],
                 config: Optional[MultiConditionConfig] = None):
        """
        Args:
            conditions: {condition_name: ConditionDataset}
            config: 训练配置
        """
        self.conditions = conditions
        self.config = config or MultiConditionConfig()

        # 初始化条件参数
        self.condition_params = {}
        for name in conditions.keys():
            self.condition_params[name] = self._init_condition_params()

        # 训练历史
        self.history = {
            'shared_loss': [],
            'condition_losses': {name: [] for name in conditions.keys()},
            'shared_params': [],
        }

    def _init_condition_params(self) -> Dict:
        """初始化单条件参数"""
        return {
            'p_move': self.config.p_move_base + np.random.uniform(-0.1, 0.1),
            'p_div': self.config.p_div_base,
            'alpha': self.config.shared_alpha,
            'beta': self.config.shared_beta,
            'edge_bonus': self.config.edge_bonus_base,
            'k_time': self.config.k_time_base,
            'gamma': self.config.shared_gamma,
            's': self.config.shared_s,
            'tau': self.config.shared_tau,
        }

    def compute_shared_loss(self,
                           condition_names: List[str],
                           objective_fn: Callable) -> float:
        """
        计算共享参数的联合损失

        Args:
            condition_names: 要训练的条件名称
            objective_fn: 目标函数

        Returns:
            shared_loss: 加权平均的损失
        """
        total_loss = 0.0
        n_conditions = len(condition_names)

        for name in condition_names:
            if name not in self.conditions:
                continue

            dataset = self.conditions[name]
            params = self.condition_params[name]

            # 使用该条件的参数和目标函数计算损失
            try:
                loss = objective_fn(
                    params=params,
                    initial_mask=dataset.initial_mask,
                    observed=dataset.observed,
                    observed_time=dataset.observed_time
                )
                total_loss += loss
                self.history['condition_losses'][name].append(loss)
            except Exception as e:
                logger.warning(f"Failed to compute loss for {name}: {e}")
                total_loss += 1e6  # 惩罚值

        return total_loss / n_conditions

    def train_shared_parameters(self,
                               condition_names: List[str],
                               objective_fn: Callable,
                               n_iterations: int = 100,
                               learning_rate: float = 0.01) -> Dict:
        """
        训练共享机制参数

        Args:
            condition_names: 参与训练的条件
            objective_fn: 目标函数
            n_iterations: 迭代次数
            learning_rate: 学习率

        Returns:
            training_results: 训练结果
        """
        best_loss = float('inf')
        best_params = None

        for it in range(n_iterations):
            # 梯度下降更新共享参数
            # 这里使用简化版本：随机扰动

            # 1. 评估当前损失
            current_loss = self.compute_shared_loss(condition_names, objective_fn)
            self.history['shared_loss'].append(current_loss)

            # 记录最佳
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = deepcopy(self.condition_params)

            # 2. 更新共享参数（所有条件同步）
            self._update_shared_params(learning_rate)

            # 3. 更新条件特异参数
            self._update_condition_params(condition_names, learning_rate)

            # 记录历史
            self.history['shared_params'].append({
                'gamma': self.config.shared_gamma,
                's': self.config.shared_s,
                'tau': self.config.shared_tau,
            })

            if it % 10 == 0:
                logger.info(f"Iteration {it}: Shared loss = {current_loss:.4f}")

        return {
            'best_loss': best_loss,
            'best_params': best_params,
            'history': self.history,
        }

    def _update_shared_params(self, learning_rate: float):
        """更新共享参数"""
        # 简化的随机梯度下降（用随机扰动模拟梯度）
        # 实际应用中应该使用真正的梯度

        delta = learning_rate * 0.1

        # Gamma更新
        if np.random.random() < 0.5:
            self.config.shared_gamma = np.clip(
                self.config.shared_gamma + np.random.uniform(-delta, delta),
                0.0, 2.0
            )

        # Tau更新（离散）
        if np.random.random() < 0.1:
            self.config.shared_tau = int(np.clip(
                self.config.shared_tau + np.random.choice([-1, 1]),
                0, 5
            ))

        # 同步到所有条件
        for name in self.condition_params:
            self.condition_params[name]['gamma'] = self.config.shared_gamma
            self.condition_params[name]['tau'] = self.config.shared_tau
            self.condition_params[name]['s'] = self.config.shared_s

    def _update_condition_params(self, condition_names: List[str], learning_rate: float):
        """更新条件特异参数"""
        delta = learning_rate * 0.1

        for name in condition_names:
            if name not in self.condition_params:
                continue

            params = self.condition_params[name]

            # p_move
            params['p_move'] = np.clip(
                params['p_move'] + np.random.uniform(-delta, delta),
                0.01, 0.99
            )

            # k_time
            params['k_time'] = np.clip(
                params['k_time'] + np.random.uniform(-delta, delta),
                0.1, 5.0
            )

            # edge_bonus
            params['edge_bonus'] = np.clip(
                params['edge_bonus'] + np.random.uniform(-delta, delta),
                0.5, 5.0
            )

    def evaluate_generalization(self,
                                train_conditions: List[str],
                                test_condition: str,
                                objective_fn: Callable) -> Dict:
        """
        评估跨条件泛化能力

        Args:
            train_conditions: 训练条件列表
            test_condition: 测试条件
            objective_fn: 目标函数

        Returns:
            generalization_results: 泛化结果
        """
        results = {
            'train_conditions': train_conditions,
            'test_condition': test_condition,
            'train_loss': 0.0,
            'test_loss': 0.0,
            'ratio': 0.0,
            'transfer_success': False,
        }

        # 计算训练集损失
        train_loss = self.compute_shared_loss(train_conditions, objective_fn)
        results['train_loss'] = train_loss

        # 计算测试集损失
        test_loss = self.compute_shared_loss([test_condition], objective_fn)
        results['test_loss'] = test_loss

        # 泛化比率
        if train_loss > 0:
            results['ratio'] = test_loss / train_loss

        # 判断是否成功（阈值：1.2倍）
        results['transfer_success'] = results['ratio'] < 1.2

        return results

    def cross_validation(self,
                        objective_fn: Callable,
                        n_folds: int = 3) -> Dict:
        """
        交叉验证评估泛化能力

        对每个条件作为测试集，其他作为训练集

        Args:
            objective_fn: 目标函数
            n_folds: 折数

        Returns:
            cv_results: 交叉验证结果
        """
        condition_names = list(self.conditions.keys())
        n_conditions = len(condition_names)

        if n_conditions < 2:
            logger.warning("Need at least 2 conditions for cross-validation")
            return {}

        all_results = []

        for test_cond in condition_names:
            train_conds = [c for c in condition_names if c != test_cond]

            # 训练
            self.train_shared_parameters(
                train_conds, objective_fn, n_iterations=50
            )

            # 评估
            result = self.evaluate_generalization(
                train_conds, test_cond, objective_fn
            )
            all_results.append(result)

        # 汇总
        avg_ratio = np.mean([r['ratio'] for r in all_results])
        success_rate = np.mean([r['transfer_success'] for r in all_results])

        return {
            'individual_results': all_results,
            'average_ratio': avg_ratio,
            'success_rate': success_rate,
            'overall_success': success_rate > 0.7 and avg_ratio < 1.3,
        }

    def get_parameter_stability(self) -> Dict:
        """
        分析参数稳定性

        检查共享参数在不同条件下的变异程度

        Returns:
            stability_metrics: 稳定性指标
        """
        shared_param_names = ['gamma', 'tau', 's']
        stability = {}

        for param_name in shared_param_names:
            values = []
            for cond_params in self.condition_params.values():
                values.append(cond_params.get(param_name, 0))

            values = np.array(values)
            stability[param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'cv': float(np.std(values) / (np.mean(values) + 1e-10)),
                'range': (float(np.min(values)), float(np.max(values))),
            }

        return stability

    def save_results(self, filepath: str):
        """保存训练结果"""
        import json
        from datetime import datetime

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': convert(self.config.__dict__),
            'condition_params': convert(self.condition_params),
            'history': convert(self.history),
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Multi-condition training results saved to {filepath}")


class ParameterTransferAnalyzer:
    """
    参数迁移能力分析器

    分析：
    1. 参数在不同条件间的迁移能力
    2. 哪些参数是条件不变的（mechanism）
    3. 哪些参数是条件依赖的（context）
    """

    def __init__(self, condition_params: Dict[str, Dict]):
        """
        Args:
            condition_params: {condition_name: {param_name: value}}
        """
        self.condition_params = condition_params
        self.param_names = list(next(iter(condition_params.values())).keys())

    def analyze_transferability(self) -> Dict:
        """分析参数可迁移性"""
        transferability = {}

        for param_name in self.param_names:
            values = []
            for cond_params in self.condition_params.values():
                values.append(cond_params.get(param_name, 0))

            values = np.array(values)

            # 变异系数
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / (abs(mean_val) + 1e-10)

            # 判断可迁移性
            if cv < 0.2:
                level = 'HIGH'  # 高度可迁移（机制参数）
            elif cv < 0.5:
                level = 'MEDIUM'  # 中等可迁移
            else:
                level = 'LOW'  # 条件依赖

            transferability[param_name] = {
                'values': values.tolist(),
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'transferability_level': level,
            }

        return transferability

    def recommend_separation(self) -> Tuple[List[str], List[str]]:
        """
        推荐参数分离策略

        Returns:
            shared_params: 应该共享的参数（机制）
            condition_params: 应该分离的参数（条件）
        """
        transfer = self.analyze_transferability()

        shared = []
        separated = []

        for param_name, metrics in transfer.items():
            if metrics['transferability_level'] == 'HIGH':
                shared.append(param_name)
            else:
                separated.append(param_name)

        return shared, separated


def plot_multi_condition_results(results: Dict, save_path: Optional[str] = None):
    """可视化多条件训练结果"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    if 'history' in results and 'shared_loss' in results['history']:
        loss_history = results['history']['shared_loss']
        ax1.plot(loss_history, label='Shared Loss', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 条件间损失对比
    ax2 = fig.add_subplot(gs[0, 1])
    if 'history' in results and 'condition_losses' in results['history']:
        condition_losses = results['history']['condition_losses']
        for name, losses in condition_losses.items():
            if losses:
                ax2.plot(losses[-50:], label=name, alpha=0.7)
        ax2.set_xlabel('Iteration (last 50)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Condition-Specific Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 泛化比率
    ax3 = fig.add_subplot(gs[1, 0])
    if 'cv_results' in results:
        cv = results['cv_results']
        individual = cv.get('individual_results', [])
        if individual:
            test_conds = [r['test_condition'] for r in individual]
            ratios = [r['ratio'] for r in individual]
            colors = ['green' if r < 1.2 else 'orange' if r < 1.5 else 'red' for r in ratios]

            ax3.barh(test_conds, ratios, color=colors)
            ax3.axvline(x=1.2, color='r', linestyle='--', label='Threshold (1.2)')
            ax3.set_xlabel('Test/Train Loss Ratio')
            ax3.set_title('Generalization Performance')
            ax3.legend()

    # 4. 参数稳定性
    ax4 = fig.add_subplot(gs[1, 1])
    if 'stability' in results:
        stability = results['stability']
        params = list(stability.keys())
        cvs = [stability[p]['cv'] for p in params]
        colors = ['green' if cv < 0.2 else 'orange' if cv < 0.5 else 'red' for cv in cvs]

        ax4.barh(params, cvs, color=colors)
        ax4.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5)
        ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Coefficient of Variation')
        ax4.set_title('Parameter Stability Across Conditions')
        ax4.set_xscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Multi-condition plot saved to {save_path}")

    return fig


if __name__ == "__main__":
    # 测试多条件训练框架
    print("Testing Multi-Condition Training Framework...")

    # 创建模拟数据集
    np.random.seed(42)

    conditions = {}
    for name in ['spontaneous', 'inhibitor', 'opto']:
        initial_mask = np.ones((100, 100))
        initial_mask[30:70, 30:70] = 0

        conditions[name] = ConditionDataset(
            name=name,
            initial_mask=initial_mask,
            observed={'A_t': np.array([1600, 800, 200])},
            observed_time=np.array([0, 17, 40]),
            metadata={'condition_type': name}
        )

    # 创建训练器
    trainer = MultiConditionTrainer(conditions)

    # 模拟训练
    def dummy_objective_fn(params, initial_mask, observed, observed_time):
        return np.random.uniform(0.1, 0.5)

    results = trainer.train_shared_parameters(
        condition_names=['spontaneous', 'inhibitor'],
        objective_fn=dummy_objective_fn,
        n_iterations=20
    )

    print("\nTraining Results:")
    print(f"  Best loss: {results['best_loss']:.4f}")

    # 评估泛化
    gen_result = trainer.evaluate_generalization(
        train_conditions=['spontaneous'],
        test_condition='inhibitor',
        objective_fn=dummy_objective_fn
    )

    print(f"\nGeneralization:")
    print(f"  Test/Train ratio: {gen_result['ratio']:.2f}")
    print(f"  Transfer success: {gen_result['transfer_success']}")

    print("\nMulti-condition training test complete!")
