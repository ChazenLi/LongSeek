"""
Causal Validation Framework (Week 8)

Implements falsifiable causal validation:
1. Optogenetic intervention prediction
2. Falsifiability test (s=+1 vs s=-1)
3. Counterfactual analysis
4. Causal error metrics

Core Principle:
A mechanistic model is falsifiable if:
- It makes testable predictions about interventions
- Wrong mechanistic assumptions can be rejected
- Correct assumptions survive validation tests
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class CausalPredictionResult:
    """因果预测结果"""
    predicted_velocity: np.ndarray
    predicted_direction: np.ndarray
    observed_velocity: Optional[np.ndarray] = None
    observed_direction: Optional[np.ndarray] = None

    # 误差指标
    velocity_error: Optional[float] = None
    direction_error: Optional[float] = None
    causal_consistency: Optional[bool] = None


@dataclass
class FalsifiabilityTestResult:
    """可证伪性测试结果"""
    loss_correct: float  # s=+1时的损失
    loss_wrong: float    # s=-1时的损失
    improvement: float   # (wrong - correct) / correct
    is_falsifiable: bool  # s=+1是否显著优于s=-1
    confidence: str      # 置信度


class CausalValidator:
    """
    因果验证器

    用于验证模型对干预效果的预测能力

    核心假设：
    - 如果s=+1（细胞沿ERK梯度迁移），则：
      - 光遗传激活局部ERK时，细胞应向激活区域迁移
    - 如果s=-1（细胞逆ERK梯度迁移），则：
      - 细胞应远离激活区域

    可证伪性：
    - 训练时用s=+1，预测opto数据
    - 再训练时用s=-1，预测opto数据
    - 如果s=+1显著优于s=-1 → 假设得到验证
    """

    def __init__(self,
                 base_params: Dict,
                 erk_field_base: Optional[np.ndarray] = None):
        """
        Args:
            base_params: 基础CA参数（从spontaneous/inhibitor学习）
            erk_field_base: 基础ERK场（用于比较）
        """
        self.base_params = base_params
        self.erk_field_base = erk_field_base

    def predict_opto_migration(self,
                               opto_erk_field: np.ndarray,
                               initial_mask: np.ndarray,
                               ca_class: Callable,
                               s_sign: int = +1) -> CausalPredictionResult:
        """
        预测光遗传干预下的迁移行为

        Args:
            opto_erk_field: opto ERK场序列 (T, H, W)
            initial_mask: 初始细胞mask
            ca_class: CA模型类
            s_sign: 符号参数（+1或-1）

        Returns:
            CausalPredictionResult: 预测结果
        """
        # 构造参数
        params = deepcopy(self.base_params)
        if 'coupling' in params:
            params['coupling']['s'] = s_sign
            params['coupling']['gamma'] = abs(params['coupling'].get('gamma', 1.0))
        else:
            params['s'] = s_sign
            params['gamma'] = abs(params.get('gamma', 1.0))

        # 创建CA实例
        H, W = opto_erk_field.shape[1], opto_erk_field.shape[2]
        ca = ca_class(H, W, params, erk_field=opto_erk_field)

        # 初始化
        ca.initialize_from_mask(initial_mask)

        # 运行模拟
        history = ca.run(num_steps=opto_erk_field.shape[0],
                        erk_field_sequence=opto_erk_field)

        # 提取预测
        predicted_velocity = self._extract_velocity(history)
        predicted_direction = self._predict_migration_direction(ca, opto_erk_field)

        return CausalPredictionResult(
            predicted_velocity=predicted_velocity,
            predicted_direction=predicted_direction,
        )

    def _extract_velocity(self, history: List[Dict]) -> np.ndarray:
        """从历史记录提取迁移速度"""
        velocities = []
        for h in history:
            velocities.append(h.get('migrations', 0))
        return np.array(velocities)

    def _predict_migration_direction(self,
                                    ca,
                                    erk_field: np.ndarray) -> np.ndarray:
        """预测迁移方向"""
        T = erk_field.shape[0]
        directions = np.zeros((T, 2))

        for t in range(T):
            if hasattr(ca, 'grad_y') and hasattr(ca, 'grad_x'):
                # 平均梯度方向作为预测方向
                grad_y = ca.grad_y if ca.grad_y is not None else 0
                grad_x = ca.grad_x if ca.grad_x is not None else 0

                # 归一化
                norm = np.sqrt(grad_y**2 + grad_x**2) + 1e-10
                directions[t] = [grad_y / norm, grad_x / norm]

        return directions

    def compute_causal_errors(self,
                             prediction: CausalPredictionResult,
                             observed_velocity: np.ndarray,
                             observed_direction: np.ndarray) -> CausalPredictionResult:
        """
        计算因果误差

        Args:
            prediction: 模型预测
            observed_velocity: 观测到的迁移速度
            observed_direction: 观测到的迁移方向

        Returns:
            更新后的预测结果（包含误差）
        """
        # 速度误差（归一化）
        pred_v_norm = prediction.predicted_velocity / (prediction.predicted_velocity[0] + 1e-10)
        obs_v_norm = observed_velocity / (observed_velocity[0] + 1e-10)
        velocity_error = np.mean(np.abs(pred_v_norm - obs_v_norm))

        # 方向误差（角度）
        direction_error = self._compute_direction_error(
            prediction.predicted_direction,
            observed_direction
        )

        # 因果一致性判断
        causal_consistency = (
            velocity_error < 0.3 and  # 30%误差内
            direction_error < np.radians(25)  # 25度内
        )

        prediction.observed_velocity = observed_velocity
        prediction.observed_direction = observed_direction
        prediction.velocity_error = velocity_error
        prediction.direction_error = direction_error
        prediction.causal_consistency = causal_consistency

        return prediction

    def _compute_direction_error(self,
                                 pred_dir: np.ndarray,
                                 obs_dir: np.ndarray) -> float:
        """计算方向误差（平均角度差）"""
        errors = []
        for p, o in zip(pred_dir, obs_dir):
            # 点积
            dot = np.clip(np.dot(p, o), -1.0, 1.0)
            angle = np.arccos(dot)
            errors.append(angle)

        return np.mean(errors) if errors else 0.0

    def falsifiability_test(self,
                           initial_mask: np.ndarray,
                           opto_erk_field: np.ndarray,
                           ca_class: Callable,
                           objective_fn: Callable,
                           n_trials: int = 5) -> FalsifiabilityTestResult:
        """
        可证伪性测试

        对比s=+1和s=-1的预测能力

        Args:
            initial_mask: 初始mask
            opto_erk_field: opto ERK场
            ca_class: CA类
            objective_fn: 目标函数
            n_trials: 试验次数

        Returns:
            FalsifiabilityTestResult: 测试结果
        """
        losses_correct = []
        losses_wrong = []

        for _ in range(n_trials):
            # 测试s=+1（正确假设）
            pred_correct = self.predict_opto_migration(
                opto_erk_field, initial_mask, ca_class, s_sign=+1
            )
            loss_correct = objective_fn(pred_correct)
            losses_correct.append(loss_correct)

            # 测试s=-1（错误假设）
            pred_wrong = self.predict_opto_migration(
                opto_erk_field, initial_mask, ca_class, s_sign=-1
            )
            loss_wrong = objective_fn(pred_wrong)
            losses_wrong.append(loss_wrong)

        avg_loss_correct = np.mean(losses_correct)
        avg_loss_wrong = np.mean(losses_wrong)
        improvement = (avg_loss_wrong - avg_loss_correct) / (avg_loss_correct + 1e-10)

        # 判断可证伪性：错误假设显著更差
        is_falsifiable = improvement > 0.1  # 10%改善

        # 置信度
        if improvement > 0.5:
            confidence = 'HIGH'
        elif improvement > 0.2:
            confidence = 'MEDIUM'
        elif improvement > 0.1:
            confidence = 'LOW'
        else:
            confidence = 'NONE'

        return FalsifiabilityTestResult(
            loss_correct=avg_loss_correct,
            loss_wrong=avg_loss_wrong,
            improvement=improvement,
            is_falsifiable=is_falsifiable,
            confidence=confidence
        )

    def counterfactual_analysis(self,
                               initial_mask: np.ndarray,
                               base_erk_field: np.ndarray,
                               ca_class: Callable,
                               intervention: str = 'remove_ERK') -> Dict:
        """
        反事实分析

        问：如果移除ERK信号，伤口愈合会怎样？

        Args:
            initial_mask: 初始mask
            base_erk_field: 基础ERK场
            ca_class: CA类
            intervention: 干预类型

        Returns:
            analysis_results: 分析结果
        """
        results = {
            'baseline': {},
            'counterfactual': {},
            'difference': {},
        }

        # 1. 基线：有ERK场
        H, W = base_erk_field.shape[1], base_erk_field.shape[2]
        ca_base = ca_class(H, W, self.base_params, erk_field=base_erk_field)
        ca_base.initialize_from_mask(initial_mask)
        history_base = ca_base.run(num_steps=base_erk_field.shape[0],
                                   erk_field_sequence=base_erk_field)

        results['baseline'] = {
            'final_wound_area': history_base[-1]['wound_area'],
            'total_migrations': sum(h['migrations'] for h in history_base),
            'total_divisions': sum(h['divisions'] for h in history_base),
        }

        # 2. 反事实：无ERK场（移除耦合）
        params_no_ERK = deepcopy(self.base_params)
        if 'coupling' in params_no_ERK:
            params_no_ERK['coupling']['gamma'] = 0.0
        else:
            params_no_ERK['gamma'] = 0.0

        ca_counter = ca_class(H, W, params_no_ERK, erk_field=None)
        ca_counter.initialize_from_mask(initial_mask)
        history_counter = ca_counter.run(num_steps=base_erk_field.shape[0])

        results['counterfactual'] = {
            'final_wound_area': history_counter[-1]['wound_area'],
            'total_migrations': sum(h['migrations'] for h in history_counter),
            'total_divisions': sum(h['divisions'] for h in history_counter),
        }

        # 3. 差异
        results['difference'] = {
            'wound_area_diff': results['counterfactual']['final_wound_area'] -
                               results['baseline']['final_wound_area'],
            'migration_diff': results['counterfactual']['total_migrations'] -
                               results['baseline']['total_migrations'],
        }

        return results


class CausalMetrics:
    """因果误差指标计算"""

    @staticmethod
    def causal_intervention_error(simulated: Dict,
                                 observed: Dict,
                                 intervention_type: str) -> float:
        """
        计算干预误差

        Args:
            simulated: 模拟结果
            observed: 观测数据
            intervention_type: 干预类型

        Returns:
            error: 误差值
        """
        # 根据干预类型计算不同误差
        if intervention_type == 'opto_activation':
            # 光遗传激活：检查方向
            return CausalMetrics._direction_alignment_error(simulated, observed)
        elif intervention_type == 'inhibitor':
            # 抑制剂：检查速度变化
            return CausalMetrics._velocity_change_error(simulated, observed)
        else:
            return 0.0

    @staticmethod
    def _direction_alignment_error(simulated: Dict, observed: Dict) -> float:
        """方向对齐误差"""
        # 获取梯度方向
        sim_grad = simulated.get('mean_gradient_direction', 0.0)
        obs_grad = observed.get('mean_gradient_direction', 0.0)

        # 角度差
        angle_diff = abs(sim_grad - obs_grad)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # 处理周期性

        return float(angle_diff)

    @staticmethod
    def _velocity_change_error(simulated: Dict, observed: Dict) -> float:
        """速度变化误差"""
        sim_v = simulated.get('average_velocity', 0.0)
        obs_v = observed.get('average_velocity', 0.0)

        # 归一化误差
        return abs(sim_v - obs_v) / (obs_v + 1e-10)

    @staticmethod
    def compute_causal_r2(simulated: np.ndarray,
                          observed: np.ndarray,
                          baseline: Optional[np.ndarray] = None) -> Dict:
        """
        计算因果R²

        衡量模型对干预的预测能力

        Args:
            simulated: 模拟值
            observed: 观测值
            baseline: 基线模型预测（可选）

        Returns:
            metrics: {r2, explained_variance}
        """
        # 总方差
        ss_tot = np.sum((observed - np.mean(observed))**2)

        # 残差平方和
        ss_res = np.sum((observed - simulated)**2)

        # R²
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # 解释方差
        explained_variance = 1 - np.var(observed - simulated) / (np.var(observed) + 1e-10)

        return {
            'r2': float(r2),
            'explained_variance': float(explained_variance),
        }


def plot_causal_validation(results: Dict, save_path: Optional[str] = None):
    """可视化因果验证结果"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 1. 可证伪性测试结果
    ax1 = fig.add_subplot(gs[0, 0])
    if 'falsifiability' in results:
        fals = results['falsifiability']
        models = ['s=+1\n(Correct)', 's=-1\n(Wrong)']
        losses = [fals['loss_correct'], fals['loss_wrong']]
        colors = ['green' if fals['is_falsifiable'] else 'red',
                 'orange' if fals['is_falsifiable'] else 'red']

        ax1.bar(models, losses, color=colors, alpha=0.7)
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Falsifiability Test ({fals["confidence"]} Confidence)')
        ax1.text(0.5, fals['loss_correct'] * 1.05,
                f'{fals["improvement"]*100:.1f}% better',
                ha='center', fontsize=10, fontweight='bold')

    # 2. 方向预测 vs 观测
    ax2 = fig.add_subplot(gs[0, 1])
    if 'direction_comparison' in results:
        comp = results['direction_comparison']
        time_points = range(len(comp['predicted']))

        ax2.plot(time_points, comp['predicted'], 'g-', label='Predicted (s=+1)', linewidth=2)
        ax2.plot(time_points, comp['observed'], 'b--', label='Observed', linewidth=2)
        ax2.fill_between(time_points,
                         comp['predicted'] - comp['std'],
                         comp['predicted'] + comp['std'],
                         color='green', alpha=0.2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Direction (rad)')
        ax2.set_title('Migration Direction Prediction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 反事实分析
    ax3 = fig.add_subplot(gs[1, 0])
    if 'counterfactual' in results:
        cf = results['counterfactual']

        metrics = ['Wound Area', 'Migrations', 'Divisions']
        baseline = [
            cf['baseline']['final_wound_area'],
            cf['baseline']['total_migrations'] / 100,
            cf['baseline']['total_divisions'] / 10,
        ]
        counterfactual = [
            cf['counterfactual']['final_wound_area'],
            cf['counterfactual']['total_migrations'] / 100,
            cf['counterfactual']['total_divisions'] / 10,
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax3.bar(x - width/2, baseline, width, label='With ERK', color='blue', alpha=0.7)
        ax3.bar(x + width/2, counterfactual, width, label='Without ERK', color='red', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel('Value (normalized)')
        ax3.set_title('Counterfactual: ERK Ablation')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    # 4. 因果误差汇总
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    text = "CAUSAL VALIDATION SUMMARY\n\n"

    if 'falsifiability' in results:
        fals = results['falsifiability']
        text += f"Falsifiability Test:\n"
        text += f"  Status: {'✅ PASS' if fals['is_falsifiable'] else '❌ FAIL'}\n"
        text += f"  Confidence: {fals['confidence']}\n"
        text += f"  Improvement: {fals['improvement']*100:.1f}%\n\n"

    if 'direction_error' in results:
        text += f"Direction Error: {results['direction_error']:.2f} rad\n"

    if 'velocity_error' in results:
        text += f"Velocity Error: {results['velocity_error']*100:.1f}%\n\n"

    text += "Conclusion:\n"
    if 'falsifiability' in results and results['falsifiability']['is_falsifiable']:
        text += "  ✅ Model predictions are causally valid.\n"
        text += "  ✅ s=+1 hypothesis is supported by data.\n"
    else:
        text += "  ⚠️ Model predictions lack causal validation.\n"
        text += "  ⚠️ Consider revising mechanistic assumptions.\n"

    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Causal validation plot saved to {save_path}")

    return fig


if __name__ == "__main__":
    # 测试因果验证框架
    print("Testing Causal Validation Framework...")

    # 模拟参数
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

    # 创建模拟ERK场
    H, W, T = 50, 50, 10
    erk_field = np.zeros((T, H, W))
    for t in range(T):
        x = np.linspace(-3, 3, W)
        y = np.linspace(-3, 3, H)
        X, Y = np.meshgrid(x, y)
        erk_field[t] = 1.0 / (1 + 0.5 * (X**2 + Y**2))

    # 创建验证器
    validator = CausalValidator(base_params, erk_field)

    # 模拟预测
    initial_mask = np.ones((H, W))
    initial_mask[20:30, 20:30] = 0

    print("\nSimulating predictions...")
    pred_correct = validator.predict_opto_migration(
        erk_field, initial_mask, lambda *args: None, s_sign=+1
    )

    pred_wrong = validator.predict_opto_migration(
        erk_field, initial_mask, lambda *args: None, s_sign=-1
    )

    print(f"  Predicted velocity (s=+1): {pred_correct.predicted_velocity[:3]}")
    print(f"  Predicted velocity (s=-1): {pred_wrong.predicted_velocity[:3]}")

    # 反事实分析
    print("\nCounterfactual analysis...")
    class MockCA:
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
                history.append({'wound_area': 1000 - i*50, 'migrations': 100, 'divisions': 10})
            return history

    cf_result = validator.counterfactual_analysis(
        initial_mask, erk_field, MockCA
    )

    print(f"  Baseline wound area: {cf_result['baseline']['final_wound_area']}")
    print(f"  Counterfactual wound area: {cf_result['counterfactual']['final_wound_area']}")
    print(f"  Difference: {cf_result['difference']['wound_area_diff']:.1f}")

    print("\nCausal validation framework test complete!")
