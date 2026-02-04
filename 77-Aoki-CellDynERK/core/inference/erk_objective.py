"""
ERK-Coupling Multi-Objective Loss Function (重构版)

多目标损失函数: L = w_A*L_A + w_rough*L_rough + w_angle*L_angle + w_flow*L_flow + w_front*L_front

权重设计:
- w_A: 低 (0.1) - A(t)变化很小，只是baseline
- w_rough: 中 (0.5) - 形态约束
- w_angle: 高 (1.0) - 核心耦合指标 ∠(v, ∇C)
- w_flow: 中 (0.3) - 光流约束（如果有）
- w_front: 中 (0.5) - 前沿位置

作者: IF-CA Team
日期: 2026-02-04
版本: 2.0 (Refactored)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)


@dataclass
class LossWeights:
    """多目标损失权重配置"""
    L_A: float = 0.1          # 面积损失权重 (低)
    L_rough: float = 0.5      # 粗糙度损失权重 (中)
    L_angle: float = 1.0      # 角度耦合损失权重 (高 - 核心!)
    L_flow: float = 0.3       # 光流损失权重 (中)
    L_front: float = 0.5      # 前沿损失权重 (中)

    def to_dict(self) -> Dict[str, float]:
        return {
            'L_A': self.L_A,
            'L_rough': self.L_rough,
            'L_angle': self.L_angle,
            'L_flow': self.L_flow,
            'L_front': self.L_front
        }


@dataclass
class LossResult:
    """损失计算结果"""
    total: float                          # 总损失
    L_A: float = 0.0                      # 面积损失
    L_rough: float = 0.0                  # 粗糙度损失
    L_angle: float = 0.0                  # 角度耦合损失
    L_flow: float = 0.0                   # 光流损失
    L_front: float = 0.0                  # 前沿损失
    weights: LossWeights = field(default_factory=LossWeights)  # 使用的权重
    details: Dict[str, Any] = field(default_factory=dict)      # 额外详情

    def to_dict(self) -> Dict[str, float]:
        """转换为字典（用于日志记录）"""
        return {
            'total': self.total,
            'L_A': self.L_A,
            'L_rough': self.L_rough,
            'L_angle': self.L_angle,
            'L_flow': self.L_flow,
            'L_front': self.L_front,
            **self.details
        }


def calculate_area_loss(
    sim_A: np.ndarray,
    obs_A: np.ndarray,
    sim_time: np.ndarray,
    obs_time: np.ndarray
) -> float:
    """
    L_A: 面积损失

    Args:
        sim_A: 模拟面积序列
        obs_A: 观测面积序列
        sim_time: 模拟时间点
        obs_time: 观测时间点

    Returns:
        MSE损失
    """
    # 归一化
    sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A
    obs_A_norm = obs_A / obs_A[0] if obs_A[0] > 0 else obs_A

    # 插值到观测时间点
    if len(sim_time) > 1:
        f_sim = interp1d(sim_time, sim_A_norm, kind='linear',
                        bounds_error=False, fill_value="extrapolate")
        sim_at_obs = f_sim(obs_time)
    else:
        sim_at_obs = np.full_like(obs_A_norm, sim_A_norm[0])

    # MSE
    mse = np.mean((sim_at_obs - obs_A_norm) ** 2)
    return mse


def calculate_roughness_loss(
    sim_grid: np.ndarray,
    obs_roughness: np.ndarray,
    sim_time: np.ndarray,
    obs_time: np.ndarray
) -> float:
    """
    L_rough: 粗糙度损失

    使用距离场梯度作为粗糙度度量

    Args:
        sim_grid: 模拟网格 (H, W)
        obs_roughness: 观测粗糙度序列
        sim_time: 模拟时间点
        obs_time: 观测时间点

    Returns:
        MSE损失
    """
    # 计算模拟粗糙度（基于距离场）
    wound_mask = (sim_grid == 0).astype(np.float32)
    distance_field = distance_transform_edt(wound_mask)

    # 梯度幅度作为粗糙度
    grad_y, grad_x = np.gradient(distance_field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 在伤口区域内平均
    if distance_field.max() > 0:
        sim_roughness = np.mean(grad_mag[distance_field > 0])
    else:
        sim_roughness = 0.0

    # 如果观测是序列，取最后一个值比较
    if len(obs_roughness) > 0:
        obs_rough = obs_roughness[-1] if len(obs_roughness) == 1 else obs_roughness
    else:
        obs_rough = np.array([1.0])

    # MSE
    mse = np.mean((sim_roughness - obs_rough) ** 2)
    return mse


def calculate_angle_coupling_loss(
    sim_angle_v_grad_C: np.ndarray,
    obs_angle_v_grad_C: np.ndarray,
    sim_time: np.ndarray,
    obs_time: np.ndarray
) -> float:
    """
    L_angle: 角度耦合损失 - 核心指标!

    ∠(v, ∇C) - 迁移方向与ERK梯度的角度
    这是最重要的观测，直接反映ERK-细胞耦合机制

    Args:
        sim_angle_v_grad_C: 模拟的角度序列 (弧度)
        obs_angle_v_grad_C: 观测的角度序列 (弧度)
        sim_time: 模拟时间点
        obs_time: 观测时间点

    Returns:
        MSE损失（角度空间）
    """
    # 处理输入
    if len(sim_angle_v_grad_C) == 0:
        return 0.0

    # 插值到观测时间点
    if len(sim_time) > 1 and len(sim_angle_v_grad_C) > 1:
        f_sim = interp1d(sim_time, sim_angle_v_grad_C, kind='linear',
                        bounds_error=False, fill_value="extrapolate")
        sim_at_obs = f_sim(obs_time)
    else:
        sim_at_obs = np.full_like(obs_angle_v_grad_C,
                                   np.mean(sim_angle_v_grad_C))

    # 角度差损失（考虑周期性）
    angle_diff = np.abs(sim_at_obs - obs_angle_v_grad_C)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # 处理周期性

    mse = np.mean(angle_diff ** 2)
    return mse


def calculate_flow_loss(
    sim_velocity: np.ndarray,
    obs_velocity: np.ndarray,
    sim_time: np.ndarray,
    obs_time: np.ndarray
) -> float:
    """
    L_flow: 光流损失

    比较模拟速度场与观测光流

    Args:
        sim_velocity: 模拟速度 (T-1,) 或 (T, 2)
        obs_velocity: 观测光流 (T, H, W, 2) 或简化版本
        sim_time: 模拟时间点
        obs_time: 观测时间点

    Returns:
        MSE损失
    """
    # 简化版本：比较速度幅度
    if sim_velocity.ndim == 2:
        sim_speed = np.sqrt(sim_velocity[:, 0]**2 + sim_velocity[:, 1]**2)
    else:
        sim_speed = np.abs(sim_velocity)

    if obs_velocity.ndim == 4:
        # (T, H, W, 2) -> (T,) 平均速度
        obs_speed = np.sqrt(obs_velocity[..., 0]**2 + obs_velocity[..., 1]**2)
        obs_speed = obs_speed.mean(axis=(1, 2))
    else:
        obs_speed = np.abs(obs_velocity)

    # 插值
    if len(sim_time) > 1:
        f_sim = interp1d(sim_time, sim_speed, kind='linear',
                        bounds_error=False, fill_value="extrapolate")
        sim_at_obs = f_sim(obs_time)
    else:
        sim_at_obs = np.full_like(obs_speed, sim_speed[0])

    mse = np.mean((sim_at_obs - obs_speed) ** 2)
    return mse


def calculate_front_loss(
    sim_grid: np.ndarray,
    obs_front_distance: np.ndarray,
    sim_time: np.ndarray,
    obs_time: np.ndarray
) -> float:
    """
    L_front: 前沿损失

    比较前沿位置分布

    Args:
        sim_grid: 模拟网格 (H, W)
        obs_front_distance: 观测前沿距离
        sim_time: 模拟时间点
        obs_time: 观测时间点

    Returns:
        MSE损失
    """
    # 计算模拟前沿距离
    wound_mask = (sim_grid == 0).astype(np.float32)
    distance_field = distance_transform_edt(wound_mask)

    # 最大前沿距离
    if distance_field.max() > 0:
        sim_front_max = distance_field.max()
        sim_front_mean = np.mean(distance_field[distance_field > 0])
    else:
        sim_front_max = 0.0
        sim_front_mean = 0.0

    # 观测值
    if isinstance(obs_front_distance, np.ndarray):
        if obs_front_distance.ndim == 0:
            obs_front_max = float(obs_front_distance)
            obs_front_mean = float(obs_front_distance)
        else:
            obs_front_max = float(obs_front_distance.max())
            obs_front_mean = float(obs_front_distance.mean())
    else:
        obs_front_max = float(obs_front_distance)
        obs_front_mean = float(obs_front_distance)

    # MSE
    mse = ((sim_front_max - obs_front_max) ** 2 +
           (sim_front_mean - obs_front_mean) ** 2) / 2
    return mse


def multi_objective_loss(
    simulated: List[Dict[str, float]],
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    erk_field: Optional[np.ndarray] = None,
    weights: Optional[LossWeights] = None
) -> LossResult:
    """
    多目标损失函数 - ERK耦合版本

    L = w_A * L_A + w_rough * L_rough + w_angle * L_angle
        + w_flow * L_flow + w_front * L_front

    Args:
        simulated: 模拟历史记录 (List of Dict with 'grid', 'wound_area', etc.)
        observed: 观测数据字典
            - A_t: 面积时间序列
            - roughness: 粗糙度时间序列
            - angle_v_grad_C: ∠(v, ∇C) 角度时间序列
            - velocity: 速度时间序列
            - front_distance: 前沿距离
        observed_time: 观测时间点
        erk_field: ERK场 (用于计算梯度)
        weights: 损失权重

    Returns:
        LossResult对象
    """
    if weights is None:
        weights = LossWeights()  # 默认权重

    result = LossResult(weights=weights)

    # 提取模拟数据
    num_steps = len(simulated)
    sim_time = np.arange(num_steps)
    sim_A = np.array([s.get('wound_area', 0) for s in simulated])

    # 提取最后一个网格用于形态比较
    final_grid = None
    if 'grid' in simulated[-1]:
        final_grid = simulated[-1]['grid']
    elif 'wound_area' in simulated[-1]:
        # 如果没有网格，从面积估计
        pass

    # 1. 面积损失
    if 'A_t' in observed and len(observed['A_t']) > 0:
        result.L_A = calculate_area_loss(
            sim_A, observed['A_t'], sim_time, observed_time
        )

    # 2. 粗糙度损失
    if 'roughness' in observed and final_grid is not None:
        result.L_rough = calculate_roughness_loss(
            final_grid, observed['roughness'], sim_time, observed_time
        )

    # 3. 角度耦合损失 - 核心!
    if 'angle_v_grad_C' in observed:
        # 计算模拟的角度耦合
        sim_angles = _compute_simulated_angle_coupling(
            simulated, erk_field
        )
        result.L_angle = calculate_angle_coupling_loss(
            sim_angles, observed['angle_v_grad_C'], sim_time, observed_time
        )
        result.details['angle_correlation'] = _compute_angle_correlation(
            sim_angles, observed['angle_v_grad_C']
        )

    # 4. 光流损失
    if 'velocity' in observed:
        sim_velocity = _compute_simulated_velocity(simulated)
        result.L_flow = calculate_flow_loss(
            sim_velocity, observed['velocity'], sim_time, observed_time
        )

    # 5. 前沿损失
    if 'front_distance' in observed and final_grid is not None:
        result.L_front = calculate_front_loss(
            final_grid, observed['front_distance'], sim_time, observed_time
        )

    # 总损失
    result.total = (
        weights.L_A * result.L_A +
        weights.L_rough * result.L_rough +
        weights.L_angle * result.L_angle +
        weights.L_flow * result.L_flow +
        weights.L_front * result.L_front
    )

    return result


def _compute_simulated_angle_coupling(
    simulated: List[Dict],
    erk_field: Optional[np.ndarray]
) -> np.ndarray:
    """计算模拟的角度耦合 ∠(v, ∇C)"""
    if erk_field is None:
        # 无ERK场时返回零
        return np.zeros(len(simulated))

    angles = []
    grad_y, grad_x = np.gradient(erk_field, axis=(1, 2))

    for i, step in enumerate(simulated):
        # 估计迁移方向（使用质心位移）
        if i > 0 and 'wound_center' in step and 'wound_center' in simulated[i-1]:
            dy = step['wound_center'][0] - simulated[i-1]['wound_center'][0]
            dx = step['wound_center'][1] - simulated[i-1]['wound_center'][1]

            v_norm = np.sqrt(dy**2 + dx**2)
            if v_norm > 1e-10:
                vy_norm = dy / v_norm
                vx_norm = dx / v_norm

                # ERK梯度方向（空间平均）
                t_idx = min(i, grad_y.shape[0] - 1)
                gy_mean = grad_y[t_idx].mean()
                gx_mean = grad_x[t_idx].mean()
                g_norm = np.sqrt(gy_mean**2 + gx_mean**2)

                if g_norm > 1e-10:
                    gy_norm = gy_mean / g_norm
                    gx_norm = gx_mean / g_norm

                    # cos(∠)
                    cos_angle = vy_norm * gy_norm + vx_norm * gx_norm
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                    continue

        angles.append(0.0)

    return np.array(angles)


def _compute_simulated_velocity(simulated: List[Dict]) -> np.ndarray:
    """计算模拟速度"""
    velocities = []

    for i in range(1, len(simulated)):
        if 'wound_area' in simulated[i] and 'wound_area' in simulated[i-1]:
            dA = simulated[i]['wound_area'] - simulated[i-1]['wound_area']
            velocities.append(dA)
        else:
            velocities.append(0.0)

    # 填充第一个
    if velocities:
        velocities.insert(0, velocities[0])

    return np.array(velocities)


def _compute_angle_correlation(
    sim_angles: np.ndarray,
    obs_angles: np.ndarray
) -> float:
    """计算角度相关性"""
    if len(sim_angles) == 0 or len(obs_angles) == 0:
        return 0.0

    # 使用cos相似度
    sim_cos = np.cos(sim_angles)
    obs_cos = np.cos(obs_angles)

    # 插值到相同长度
    if len(sim_cos) != len(obs_cos):
        from scipy.interpolate import interp1d
        if len(sim_cos) > 1:
            f = interp1d(np.arange(len(sim_cos)), sim_cos, kind='linear')
            sim_cos = f(np.linspace(0, len(sim_cos)-1, len(obs_cos)))

    corr = np.corrcoef(sim_cos, obs_cos)[0, 1]
    return corr if not np.isnan(corr) else 0.0


# 便捷函数
def calculate_loss(
    simulated: List[Dict[str, float]],
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    erk_field: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    便捷损失计算函数

    Returns:
        (total_loss, loss_dict)
    """
    if weights is not None:
        weight_obj = LossWeights(**weights)
    else:
        weight_obj = LossWeights()

    result = multi_objective_loss(
        simulated, observed, observed_time, erk_field, weight_obj
    )

    return result.total, result.to_dict()


if __name__ == "__main__":
    # 测试多目标损失
    print("Testing ERK Multi-Objective Loss Function")

    # 创建模拟数据
    num_steps = 50
    simulated = [
        {'wound_area': 1600 - i * 20, 'wound_center': (50, 50)}
        for i in range(num_steps)
    ]

    # 创建观测数据
    observed_time = np.array([0, 0.5, 1.0, 1.5, 2.0])
    observed = {
        'A_t': np.array([1600, 1200, 800, 500, 200]),
        'roughness': np.array([1.0, 0.9, 0.8, 0.7, 0.6]),
        'angle_v_grad_C': np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
        'velocity': np.array([20, 15, 10, 8, 5]),
        'front_distance': np.array([30, 25, 20, 15, 10])
    }

    # 计算损失
    result = multi_objective_loss(simulated, observed, observed_time)

    print("\n=== Loss Result ===")
    print(f"Total Loss: {result.total:.6f}")
    print(f"  L_A (area): {result.L_A:.6f} (weight={result.weights.L_A})")
    print(f"  L_rough: {result.L_rough:.6f} (weight={result.weights.L_rough})")
    print(f"  L_angle: {result.L_angle:.6f} (weight={result.weights.L_angle})")
    print(f"  L_flow: {result.L_flow:.6f} (weight={result.weights.L_flow})")
    print(f"  L_front: {result.L_front:.6f} (weight={result.weights.L_front})")

    if 'angle_correlation' in result.details:
        print(f"\nAngle Correlation: {result.details['angle_correlation']:.4f}")

    print("\n✅ ERK Multi-Objective Loss test complete!")
