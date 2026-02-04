"""
预处理模块 - 规范化和重采样

参考 plan.md 第 4 节：
- 统一空间尺度（建议重采样到 1.0 µm/px）
- 统一时间尺度（tau 用秒而非帧）
- ERK 场规范化
"""

import numpy as np
from scipy import ndimage
from skimage.transform import resize
from typing import Tuple, Optional


def normalize_erk_field(C: np.ndarray, method: str = "robust") -> np.ndarray:
    """
    ERK 场规范化

    参考 plan.md 4.3:
    - NaN/inf 清理
    - per-frame robust normalization
    - 轻微空间平滑

    Args:
        C: (T, H, W) ERK 活性场
        method: 规范化方法，'robust' 或 'minmax'

    Returns:
        规范化后的 (T, H, W) 数组
    """
    C = C.astype(np.float32)

    # 清理 NaN 和 inf
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-frame 规范化
    if method == "robust":
        # 使用 median 和 IQR（更鲁棒）
        for t in range(C.shape[0]):
            frame = C[t]
            median = np.median(frame)
            q75 = np.percentile(frame, 75)
            q25 = np.percentile(frame, 25)
            iqr = q75 - q25 + 1e-8
            C[t] = (frame - median) / (iqr * 1.35)  # 1.35 使其接近标准差
    elif method == "minmax":
        # 简单 minmax
        for t in range(C.shape[0]):
            frame = C[t]
            min_val = frame.min()
            max_val = frame.max()
            if max_val > min_val:
                C[t] = (frame - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown method: {method}")

    return C


def smooth_spatial(C: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    空间平滑（用于求梯度前的降噪）

    Args:
        C: (T, H, W) ERK 场
        sigma: 高斯核标准差（像素）

    Returns:
        平滑后的数组
    """
    C_smooth = np.empty_like(C)
    for t in range(C.shape[0]):
        C_smooth[t] = ndimage.gaussian_filter(C[t], sigma=sigma)
    return C_smooth


def compute_gradient(C: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ERK 场的空间梯度

    Args:
        C: (T, H, W) ERK 场
        sigma: 平滑核大小

    Returns:
        grad_mag: (T, H, W) 梯度幅值
        grad_x: (T, H, W) x 方向梯度
        grad_y: (T, H, W) y 方向梯度
    """
    # 先平滑
    C_smooth = smooth_spatial(C, sigma)

    # 计算梯度
    grad_y, grad_x = np.gradient(C_smooth, axis=(1, 2))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    return grad_mag, grad_x, grad_y


def rescale_to_um(
    C: np.ndarray,
    um_per_px_original: float,
    um_per_px_target: float = 1.0,
    order: int = 1
) -> np.ndarray:
    """
    重采样到目标物理分辨率

    参考 plan.md 4.1: 统一空间尺度

    Args:
        C: (T, H, W) 输入数组
        um_per_px_original: 原始像素尺度（µm/px）
        um_per_px_target: 目标像素尺度（µm/px）
        order: 插值阶数（0=nearest, 1=bilinear, 3=cubic）

    Returns:
        重采样后的 (T, H_new, W_new) 数组
    """
    if um_per_px_original == um_per_px_target:
        return C

    scale_factor = um_per_px_original / um_per_px_target
    new_shape = (
        C.shape[0],
        int(C.shape[1] * scale_factor),
        int(C.shape[2] * scale_factor)
    )

    C_rescaled = np.empty(new_shape, dtype=C.dtype)
    for t in range(C.shape[0]):
        C_rescaled[t] = resize(
            C[t],
            (new_shape[1], new_shape[2]),
            order=order,
            preserve_range=True,
            anti_aliasing=order > 0
        ).astype(C.dtype)

    return C_rescaled


def rescale_mask(
    mask: np.ndarray,
    um_per_px_original: float,
    um_per_px_target: float = 1.0,
    threshold: float = 0.5
) -> np.ndarray:
    """
    重采样二值掩膜（保持二值）

    Args:
        mask: (T, H, W) 二值掩膜
        um_per_px_original: 原始像素尺度
        um_per_px_target: 目标像素尺度
        threshold: 二值化阈值

    Returns:
        重采样后的二值掩膜
    """
    if um_per_px_original == um_per_px_target:
        return mask

    scale_factor = um_per_px_original / um_per_px_target
    new_shape = (
        mask.shape[0],
        int(mask.shape[1] * scale_factor),
        int(mask.shape[2] * scale_factor)
    )

    mask_rescaled = np.empty(new_shape, dtype=mask.dtype)
    for t in range(mask.shape[0]):
        # 使用双线性插值后阈值化
        resampled = resize(
            mask[t].astype(np.float32),
            (new_shape[1], new_shape[2]),
            order=1,
            preserve_range=True
        )
        mask_rescaled[t] = (resampled > threshold).astype(mask.dtype)

    return mask_rescaled


def seconds_to_frames(tau_seconds: float, dt_seconds: float) -> int:
    """
    将时滞从秒转换为帧数

    参考 plan.md 4.2: 统一时间尺度

    Args:
        tau_seconds: 时滞（秒）
        dt_seconds: 时间步长（秒）

    Returns:
        帧数
    """
    return max(1, int(round(tau_seconds / dt_seconds)))


def frames_to_seconds(tau_frames: int, dt_seconds: float) -> float:
    """
    将时滞从帧数转换为秒

    Args:
        tau_frames: 帧数
        dt_seconds: 时间步长（秒）

    Returns:
        时滞（秒）
    """
    return tau_frames * dt_seconds


def check_drift(C: np.ndarray, threshold: float = 5.0) -> Tuple[float, bool]:
    """
    检查图像漂移（简单的时间相关检查）

    Args:
        C: (T, H, W) 图像序列
        threshold: 漂移阈值（像素）

    Returns:
        max_drift: 最大漂移（像素）
        has_drift: 是否超过阈值
    """
    from skimage.registration import phase_cross_correlation

    max_drift = 0.0
    ref = C[0]

    for t in range(1, min(C.shape[0], 10)):  # 只检查前 10 帧
        try:
            shift, _, _ = phase_cross_correlation(ref, C[t], normalization=None)
            drift = np.sqrt(shift[0]**2 + shift[1]**2)
            max_drift = max(max_drift, drift)
        except:
            pass

    return max_drift, max_drift > threshold


def crop_to_content(C: np.ndarray, padding: int = 10) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """
    裁剪到实际内容区域（移除空白边缘）

    Args:
        C: (T, H, W) 图像序列
        padding: 边缘留白

    Returns:
        C_cropped: 裁剪后的数组
        slices: 裁切切片
    """
    # 计算时间平均值
    mean_frame = C.mean(axis=0)

    # 找到内容区域
    mask = mean_frame > mean_frame.mean() * 0.1

    if not mask.any():
        return C, (slice(None), slice(None), slice(None))

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 添加 padding 并裁剪
    rmin = max(0, rmin - padding)
    rmax = min(C.shape[1], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(C.shape[2], cmax + padding + 1)

    C_cropped = C[:, rmin:rmax, cmin:cmax]
    slices = (slice(None), slice(rmin, rmax), slice(cmin, cmax))

    return C_cropped, slices
