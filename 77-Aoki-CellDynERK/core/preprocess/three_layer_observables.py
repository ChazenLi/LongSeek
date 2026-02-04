"""
Three-Layer Observables for ERK-Migration Coupling Analysis

三层观测体系：
1. Macro: A(t), closure_rate, velocity
2. Morphology: roughness, front_distance, anisotropy
3. Molecular-Mechanical: ∠(v, ∇C), wave_speed, time_lag_correlation

作者: IF-CA Team
日期: 2026-02-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, gaussian_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class Layer1MacroObservables:
    """Layer 1: 宏观测"""
    time_hours: np.ndarray      # (T,) 时间点(小时)
    A_t: np.ndarray                # (T,) 覆盖面积
    A_t_normalized: np.ndarray # (T,) 归一化面积
    closure_rate: float          # 闭合速度
    velocity_estimate: np.ndarray # (T-1,) 速度估计


@dataclass
class Layer2MorphologyObservables:
    """Layer 2: 形态观测"""
    front_roughness: np.ndarray      # (T,) 前沿粗糙度
    max_front_distance: np.ndarray   # (T,) 最大前沿距离
    front_distance_std: np.ndarray   # (T,) 前沿距离标准差
    anisotropy_index: np.ndarray      # (T,) 各向异性指数
    perimeter: np.ndarray           # (T,) 周长


@dataclass
class Layer3CouplingObservables:
    """Layer 3: 分子-力学耦合观测（核心创新点）"""
    angle_v_grad_C: np.ndarray         # (T,) ∠(v, ∇C) 平均角度
    angle_v_grad_C_std: np.ndarray     # (T,) 角度标准差
    cosine_alignment: np.ndarray     # (T,) cos(∠(v, ∇C)) 均值
    wave_speed: np.ndarray             # (T,) ERK波速
    wave_direction: np.ndarray        # (T,) 波前方向
    tau_sensitivity: Dict[int, float]  # {tau: correlation} τ敏感性


def compute_three_layer_observables(
    erk_field: np.ndarray,
    mask: Optional[np.ndarray] = None,
    optic_flow: Optional[np.ndarray] = None,
    dt_seconds: float = 180.0,
    pixel_size_um: float = 0.49
) -> Dict[str, np.ndarray]:
    """
    计算三层观测

    Args:
        erk_field: (T, H, W) ERK场序列
        mask: (H, W) 细胞mask (可选，如无则从ERK阈值推断)
        optic_flow: (T, H, W, 2) 光流场 (可选)
        dt_seconds: 时间间隔(秒)
        pixel_size_um: 像素大小(微米)

    Returns:
        包含所有三层观测的字典
    """
    T, H, W = erk_field.shape

    # ===== Layer 1: 宏观 =====
    time_hours = np.arange(T) * (dt_seconds / 3600.0)

    # 计算mask（如果未提供）
    if mask is None:
        # 使用ERK场的阈值
        threshold = erk_field.mean(axis=0)
        mask = (erk_field[0] > threshold).astype(np.float32)

    # A(t): 覆盖面积（0=伤口/空）
    A_t = np.sum(mask == 0, axis=(1, 2))
    A_t_normalized = A_t / A_t[0]

    # closure_rate
    if len(A_t) > 1:
        closure_rate = (A_t[0] - A_t[-1]) / (time_hours[-1] - time_hours[0])
    else:
        closure_rate = 0.0

    # ===== Layer 2: 形态 =====
    wound_mask = (mask == 0).astype(np.float32)
    distance_field = distance_transform_edt(wound_mask)

    # 前沿粗糙度（基于距离场的梯度）
    front_roughness = []
    max_front_distance = []
    front_distance_std = []
    anisotropy_index = []

    for t in range(T):
        df_t = distance_field  # 该时刻的距离场

        if df_t.max() > 0:
            # 粗糙度：距离场的梯度标准差
            gy, gx = np.gradient(df_t)
            rough = np.std(np.sqrt(gx**2 + gy**2)[df_t > 0])
            front_roughness.append(rough)

            # 最大前沿距离
            max_dist = df_t.max()
            max_front_distance.append(max_dist)

            # 前沿距离标准差
            std_dist = np.std(df_t[df_t > 0])
            front_distance_std.append(std_dist)

            # 各向异性（使用轮廓长宽比）
            gy, gx = np.gradient(df_t)
            y_indices, x_indices = np.where(df_t > 0)
            if len(x_indices) > 0:
                width = x_indices.max() - x_indices.min()
                height = y_indices.max() - y_indices.min()
                aniso = width / (height + 1e-10)
                anisotropy_index.append(aniso)
            else:
                anisotropy_index.append(1.0)
        else:
            front_roughness.append(0.0)
            max_front_distance.append(0.0)
            front_distance_std.append(0.0)
            anisotropy_index.append(1.0)

    front_roughness = np.array(front_roughness)
    max_front_distance = np.array(max_front_distance)
    front_distance_std = np.array(front_distance_std)
    anisotropy_index = np.array(anisotropy_index)

    # ===== Layer 3: 分子-力学耦合 =====
    results = {}

    # 先存入Layer 1和2的结果
    results['time_hours'] = time_hours
    results['A_t'] = A_t
    results['A_t_normalized'] = A_t_normalized
    results['closure_rate'] = np.array([closure_rate])
    results['front_roughness'] = front_roughness
    results['max_front_distance'] = max_front_distance
    results['front_distance_std'] = front_distance_std
    results['anisotropy_index'] = anisotropy_index

    # 计算ERK梯度
    grad_y, grad_x = np.gradient(erk_field, axis=(1, 2))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 平滑梯度
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=1.0)

    # 波速估计
    wave_speed = _estimate_wave_speed(grad_mag_smooth, dt_seconds, pixel_size_um)

    # 波前方向
    wave_direction = np.arctan2(
        grad_y.mean(axis=(1, 2)),
        grad_x.mean(axis=(1, 2))
    )

    # 如果有光流，计算角度耦合
    if optic_flow is not None:
        angle_results = _compute_angle_coupling_with_flow(
            optic_flow, grad_x, grad_y, mask
        )
        results.update(angle_results)
    else:
        # 无光流时，使用前沿位移方向近似
        angle_results = _compute_angle_coupling_approximate(
            max_front_distance, grad_x, grad_y, mask
        )
        results.update(angle_results)

    results['wave_speed'] = np.array([wave_speed] * T)
    results['wave_direction'] = np.array([wave_direction] * T)

    return results


def _estimate_wave_speed(grad_mag: np.ndarray, dt_seconds: float, pixel_size_um: float) -> float:
    """估计ERK波速"""
    T = grad_mag.shape[0]

    if T < 2:
        return 0.0

    # 追踪梯度峰值
    peak_positions = []
    for t in range(T):
        argmax = np.unravel_index(grad_mag[t].argmax(), grad_mag[t].shape)
        peak_positions.append(argmax)

    peak_positions = np.array(peak_positions)

    # 计算位移
    displacements = np.sqrt(
        np.diff(peak_positions[:, 0])**2 +
        np.diff(peak_positions[:, 1])**2
    )

    # 速度 (像素/秒)
    speed_px_per_sec = np.mean(displacements) / dt_seconds

    # 转换为微米/分钟
    speed_um_per_min = speed_px_per_sec * pixel_size_um * 60

    return speed_um_per_min


def _compute_angle_coupling_with_flow(
    optic_flow: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    使用光流计算∠(v, ∇C)

    Args:
        optic_flow: (T, H, W, 2) 光流场 (vy, vx)
        grad_x, grad_y: (T, H, W) ERK梯度
        mask: (H, W) 有效区域
    """
    T, H, W = grad_x.shape

    angle_v_grad_C = []
    cosine_alignment = []

    for t in range(T):
        # 光流方向
        vy = optic_flow[t, :, 0]
        vx = optic_flow[t, :, 1]

        # 归一化
        v_norm = np.sqrt(vy**2 + vx**2)
        vy_norm = np.where(v_norm > 1e-10, vy / v_norm, 0)
        vx_norm = np.where(v_norm > 1e-10, vx / v_norm, 0)

        # ERK梯度方向
        grad_mag = np.sqrt(grad_x[t]**2 + grad_y[t]**2)
        gx_norm = np.where(grad_mag > 1e-10, grad_x[t] / grad_mag, 0)
        gy_norm = np.where(grad_mag > 1e-10, grad_y[t] / grad_mag, 0)

        # 计算cos(∠)
        cos_angle = vy_norm * gy_norm + vx_norm * gx_norm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # 只在mask区域内计算
        valid_mask = mask.flatten() > 0

        valid_cos = cos_angle.flatten()[valid_mask]
        angle_v_grad_C.append(np.arccos(valid_cos))
        cosine_alignment.append(np.mean(valid_cos))

    return {
        'angle_v_grad_C': np.array(angle_v_grad_C),
        'cosine_alignment': np.array(cosine_alignment)
    }


def _compute_angle_coupling_approximate(
    max_front_distance: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    使用前沿位移近似计算∠(v, ∇C)（无光流时）

    Args:
        max_front_distance: (T,) 每时刻的最大前沿距离
        grad_x, grad_y: (T, H, W) ERK梯度
        mask: (H, W) 有效区域
    """
    T = len(max_front_distance)

    # 估计前沿位移方向（使用质心位移）
    front_displacement = []
    direction_angles = []

    wound_mask = (mask == 0).astype(float)
    from scipy.ndimage import center_of_mass

    for t in range(T):
        # 计算伤口质心
        if wound_mask.max() > 0:
            cy, cx = center_of_mass(wound_mask)
            front_displacement.append((cy, cx))
        else:
            front_displacement.append((0, 0))

    # 计算位移方向
    angle_v = []
    angle_grad = []
    cosine_alignment = []

    for t in range(1, T):
        # 前沿位移方向
        dy = front_displacement[t][0] - front_displacement[t-1][0]
        dx = front_displacement[t][1] - front_displacement[t-1][1]

        v_norm = np.sqrt(dy**2 + dx**2)
        if v_norm > 1e-10:
            vy_norm = dy / v_norm
            vx_norm = dx / v_norm

            # ERK梯度方向（空间平均）
            gy_mean = grad_y[t].mean()
            gx_mean = grad_x[t].mean()
            g_norm = np.sqrt(gy_mean**2 + gx_mean**2)
            if g_norm > 1e-10:
                gy_norm = gy_mean / g_norm
                gx_norm = gx_mean / g_norm

                # cos(∠)
                cos_angle = vy_norm * gy_norm + vx_norm * gx_norm
                cos_angle = np.clip(cos_angle, -1.0, 1.0)

                angle = np.arccos(cos_angle)
                angle_v.append(angle)
                cosine_alignment.append(cos_angle)

    # 计算ERK梯度方向时间序列
    for t in range(T):
        gy_mean = grad_y[t].mean()
        gx_mean = grad_x[t].mean()
        angle_grad.append(np.arctan2(gy_mean, gx_mean))

    if angle_v:
        return {
            'angle_v_grad_C': np.array(angle_v),
            'cosine_alignment': np.array(cosine_alignment),
            'migration_direction_angle': np.array(angle_v)
        }
    else:
        return {
            'angle_v_grad_C': np.zeros(T),
            'cosine_alignment': np.zeros(T),
            'migration_direction_angle': np.zeros(T)
        }


def compute_tau_sensitivity(
    erk_field: np.ndarray,
    migration_angles: np.ndarray,
    tau_range: np.ndarray = np.arange(0, 6)
) -> Dict[str, np.ndarray]:
    """
    计算τ敏感性 - 不同τ值下ERK梯度与迁移角度的相关性

    这是参数可辨识性分析的关键！

    Args:
        erk_field: (T, H, W) ERK场
        migration_angles: (T,) 迁移角度（弧度）
        tau_range: 要测试的τ值范围

    Returns:
        {
            'tau_values': tau_range,
            'correlations': 每个τ对应的相关系数
        }
    """
    correlations = []

    # 计算ERK梯度时间序列
    grad_y, grad_x = np.gradient(erk_field, axis=(1, 2))

    for tau in tau_range:
        # 计算滞后τ的梯度方向角度
        lagged_angles = []
        observed_angles = []

        for t in range(tau, len(migration_angles)):
            # τ时刻前的梯度方向
            if t < grad_y.shape[0]:
                angle_grad = np.arctan2(
                    grad_y[t-tau].mean(),
                    grad_x[t-tau].mean()
                )
                lagged_angles.append(angle_grad)

                # 对应的迁移角度
                observed_angles.append(migration_angles[t])

        if lagged_angles and observed_angles:
            # 计算角度相关性（使用cos距离）
            lagged_arr = np.array(lagged_angles)
            observed_arr = np.array(observed_angles)

            # 简单相关性：cos相似度
            corr = np.mean(np.cos(lagged_arr) * np.cos(observed_arr) +
                           np.sin(lagged_arr) * np.sin(observed_arr))
            correlations.append(corr)
        else:
            correlations.append(0.0)

    return {
        'tau_values': tau_range,
        'correlations': np.array(correlations),
        'optimal_tau': tau_range[np.argmax(correlations)] if correlations else 0
    }


def extract_layer3_coupling_metrics(
    erk_field: np.ndarray,
    optic_flow: Optional[np.ndarray],
    mask: Optional[np.ndarray] = None,
    tau_range: np.ndarray = np.arange(0, 6)
) -> Dict[str, any]:
    """
    提取Layer 3分子-力学耦合的所有指标

    这是论文中最重要的观测层！

    Returns:
        包含所有耦合指标的字典，可直接用于损失函数计算
    """
    T, H, W = erk_field.shape

    # 计算ERK梯度
    grad_y, grad_x = np.gradient(erk_field, axis=(1, 2))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 平滑
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=1.0)

    # 波速
    dt_seconds = 180.0  # TODO: 从metadata读取
    pixel_size_um = 0.49
    wave_speed = _estimate_wave_speed(grad_mag_smooth, dt_seconds, pixel_size_um)

    # 波前方向
    wave_direction = np.arctan2(
        grad_y.mean(axis=(1, 2)),
        grad_x.mean(axis=(1, 2))
    )

    # τ敏感性
    tau_results = {}

    # 如果有光流
    if optic_flow is not None:
        angle_v_grad_C_list = []
        for t in range(T):
            vy = optic_flow[t, :, 0]
            vx = optic_flow[t, :, 1]
            v_mag = np.sqrt(vy**2 + vx**2)

            # ERK梯度方向
            gx = grad_x[t]
            gy = grad_y[t]

            # 计算角度
            cos_angle = (vy * gy + vx * gx) / (v_mag * np.sqrt(gx**2 + gy**2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            angle_v_grad_C_list.append(angle)

        angle_v_grad_C = np.array(angle_v_grad_C_list)

        # τ敏感性
        tau_sens = compute_tau_sensitivity(erk_field, angle_v_grad_C, tau_range)
    else:
        angle_v_grad_C = None
        tau_sens = None

    return {
        'erk_gradient_magnitude': grad_mag_smooth,
        'erk_gradient_magnitude_mean': grad_mag_smooth.mean(axis=(1, 2)),
        'wave_speed_um_per_min': wave_speed,
        'wave_direction_rad': wave_direction,
        'angle_v_grad_C_series': angle_v_grad_C,
        'tau_sensitivity': tau_sens,
        'time_points_hours': np.arange(T) * (dt_seconds / 3600),
    }


# 数据类定义
Layer1 = Layer1MacroObservables
Layer2 = Layer2MorphologyObservables
Layer3 = Layer3CouplingObservables


if __name__ == "__main__":
    # 测试
    print("Testing Three-Layer Observables...")

    # 创建模拟数据
    T, H, W = 60, 100, 100

    # ERK场（模拟波传播）
    erk_field = np.zeros((T, H, W))
    for t in range(T):
        # 波从左向右传播
        erk_field[t] = 1.0 / (1 + 0.1 * (np.arange(W) - t*2)**2 / 100))

    # Mask
    mask = np.ones((H, W))
    mask[:, :30] = 0  # 左侧是"伤口"

    # 计算观测
    obs = compute_three_layer_observables(erk_field, mask)

    print(f"Layer 1 - Macro:")
    print(f"  Time range: {obs['time_hours'][0]:.1f} - {obs['time_hours'][-1]:.1f} hours")
    print(f"  A(t): {obs['A_t'][0]:.0f} -> {obs['A_t'][-1]:.0f}")
    print(f"  Closure rate: {obs['closure_rate'][0]:.6f}")

    print(f"\nLayer 2 - Morphology:")
    print(f"  Roughness (mean): {obs['front_roughness'].mean():.4f}")
    print(f"  Max distance (mean): {obs['max_front_distance'].mean():.2f}")

    print(f"\nLayer 3 - Coupling:")
    print(f"  Wave speed: {obs['wave_speed']:.2f} um/min")
    print(f"  Wave direction: {obs['wave_direction']:.2f} rad")

    print("\n✅ Three-layer observables test complete!")
