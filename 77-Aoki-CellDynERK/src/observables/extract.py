"""
观测量提取模块

参考 plan.md 第 6 节：
- 宏观层：A(t)、closure rate
- 形态层：前沿粗糙度、推进分布
- 分子层：C_obs 统计、梯度统计、波传播方向
- 运动层：光流、角度分布
"""

import numpy as np
from scipy import ndimage
from skimage import measure, filters
from typing import Tuple, Optional, Dict


def extract_macro_observables(mask: np.ndarray, um_per_px: float) -> Dict:
    """
    宏观层观测量

    Args:
        mask: (T, H, W) 占据掩膜，1=cell, 0=wound
        um_per_px: 像素尺度

    Returns:
        宏观观测量字典
    """
    T = mask.shape[0]

    # Wound 面积时间序列（µm²）
    wound_area = np.array([(1 - mask[t]).sum() * um_per_px ** 2 for t in range(T)])

    # 闭合率（µm²/s）
    closure_rate = np.diff(wound_area)

    return {
        "wound_area_um2": wound_area.tolist(),
        "closure_rate_um2_per_sec": closure_rate.tolist(),
        "final_wound_area_um2": float(wound_area[-1]),
        "closure_fraction": float(1 - wound_area[-1] / wound_area[0]) if wound_area[0] > 0 else 0.0,
    }


def extract_front_roughness(mask: np.ndarray) -> Dict:
    """
    形态层：前沿粗糙度

    将边界转换为函数后计算粗糙度

    Args:
        mask: (T, H, W) 占据掩膜

    Returns:
        前沿粗糙度字典
    """
    T = mask.shape[0]
    roughness_values = []

    for t in range(T):
        m = mask[t]
        if m.sum() == 0:
            roughness_values.append(0.0)
            continue

        # 找到边界
        contours = measure.find_contours(m, 0.5)
        if not contours:
            roughness_values.append(0.0)
            continue

        # 使用最长轮廓
        contour = max(contours, key=len)

        # 简单粗糙度：边界长度 / 端点距离
        boundary_length = len(contour)
        endpoint_distance = np.linalg.norm(contour[0] - contour[-1])

        if endpoint_distance > 0:
            roughness = boundary_length / endpoint_distance
        else:
            roughness = boundary_length / 100  # 归一化因子

        roughness_values.append(roughness)

    return {
        "front_roughness": roughness_values,
        "mean_roughness": float(np.mean(roughness_values)),
        "final_roughness": float(roughness_values[-1]) if roughness_values else 0.0,
    }


def extract_front_propagation(
    mask: np.ndarray,
    num_samples: int = 360
) -> Dict:
    """
    前沿推进距离分布

    Args:
        mask: (T, H, W) 占据掩膜
        num_samples: 角度采样数

    Returns:
        前沿推进字典
    """
    T = mask.shape[0]

    # 计算质心
    def get_centroid(m):
        if m.sum() == 0:
            return np.array([m.shape[0] / 2, m.shape[1] / 2])
        props = measure.regionprops(m.astype(int))[0]
        return np.array(props.centroid)

    # 第一帧质心作为参考
    centroid_0 = get_centroid(mask[0])

    propagation = []
    for t in range(T):
        m = mask[t]
        if m.sum() == 0:
            propagation.append([])
            continue

        centroid = get_centroid(m)

        # 在不同角度上采样从质心到边界的距离
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        distances = []

        cy, cx = centroid
        max_dist = max(m.shape) * 1.5

        for angle in angles:
            # 沿角度采样
            for r in range(int(max_dist)):
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))

                if 0 <= y < m.shape[0] and 0 <= x < m.shape[1]:
                    if m[y, x] == 0:  # 到达 wound
                        distances.append(r)
                        break
            else:
                distances.append(max_dist)

        propagation.append(distances)

    return {
        "front_distances_per_frame": propagation,
        "num_angles": num_samples,
    }


def extract_erk_statistics(C: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """
    分子层：ERK 场统计

    Args:
        C: (T, H, W) ERK 活性场
        mask: (T, H, W) 可选的占据掩膜

    Returns:
        ERK 统计字典
    """
    T = C.shape[0]

    global_mean = []
    global_std = []
    global_median = []

    front_mean = []  # 前沿邻域的 ERK 水平

    for t in range(T):
        frame = C[t]

        # 全局统计
        global_mean.append(float(np.mean(frame)))
        global_std.append(float(np.std(frame)))
        global_median.append(float(np.median(frame)))

        # 前沿邻域统计（如果有 mask）
        if mask is not None:
            # 获取前沿区域（wound 边界附近）
            m = mask[t]
            distance = ndimage.distance_transform_edt(m)
            wound_dist = ndimage.distance_transform_edt(1 - m)

            # 前沿是 cell 和 wound 交界处
            front = (distance <= 5) & (wound_dist <= 5)
            if front.sum() > 0:
                front_mean.append(float(np.mean(frame[front])))
            else:
                front_mean.append(0.0)

    result = {
        "global_mean": global_mean,
        "global_std": global_std,
        "global_median": global_median,
    }

    if mask is not None:
        result["front_mean"] = front_mean

    return result


def extract_erk_gradient(
    C: np.ndarray,
    sigma: float = 2.0
) -> Dict:
    """
    ERK 梯度统计

    Args:
        C: (T, H, W) ERK 活性场
        sigma: 平滑核大小

    Returns:
        梯度统计字典
    """
    from src.preprocess.normalize import compute_gradient

    grad_mag, grad_x, grad_y = compute_gradient(C, sigma=sigma)

    # 梯度幅值统计
    mag_mean = [float(np.mean(grad_mag[t])) for t in range(C.shape[0])]
    mag_max = [float(np.max(grad_mag[t])) for t in range(C.shape[0])]

    # 梯度方向统计
    angles = np.arctan2(grad_y, grad_x)  # (-π, π]

    return {
        "gradient_magnitude_mean": mag_mean,
        "gradient_magnitude_max": mag_max,
        "gradient_direction_mean": [float(np.mean(angles[t])) for t in range(C.shape[0])],
    }


def estimate_wave_direction(
    C: np.ndarray,
    method: str = "phase_gradient",
    temporal_window: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计波传播方向场

    Args:
        C: (T, H, W) ERK 活性场
        method: 方法，'phase_gradient' 或 'correlation'
        temporal_window: 时间窗口大小

    Returns:
        wave_dir_y: (T, H, W) y 方向波传播方向
        wave_dir_x: (T, H, W) x 方向波传播方向
    """
    T, H, W = C.shape

    if method == "phase_gradient":
        # 使用时空梯度估计
        # k = ∇φ，其中 φ 是相位
        # 简化：使用 3D 梯度后投影

        # 平滑
        C_smooth = np.empty_like(C)
        for t in range(T):
            C_smooth[t] = filters.gaussian(C[t], sigma=2, preserve_range=True)

        # 计算时空梯度
        grad_t = np.gradient(C_smooth, axis=0)
        grad_y = np.gradient(C_smooth, axis=1)
        grad_x = np.gradient(C_smooth, axis=2)

        # 波传播方向（简化假设：平面波）
        # k · v = ω，其中 v 是相速度
        wave_dir_y = np.zeros_like(C_smooth)
        wave_dir_x = np.zeros_like(C_smooth)

        for t in range(T):
            # 避免除零
            denom = grad_t[t] ** 2 + 1e-8
            wave_dir_y[t] = -grad_y[t] / denom
            wave_dir_x[t] = -grad_x[t] / denom

        # 归一化
        mag = np.sqrt(wave_dir_y ** 2 + wave_dir_x ** 2) + 1e-8
        wave_dir_y /= mag
        wave_dir_x /= mag

    elif method == "correlation":
        # 基于时空相关的方法
        wave_dir_y = np.zeros((T, H, W))
        wave_dir_x = np.zeros((T, H, W))

        # 简化实现：使用帧间位移
        for t in range(1, T):
            from skimage.registration import phase_cross_correlation
            try:
                shift, _, _ = phase_cross_correlation(C[t-1], C[t], normalization=None)
                wave_dir_y[t] = np.full((H, W), -shift[0])
                wave_dir_x[t] = np.full((H, W), -shift[1])
            except:
                pass

    else:
        raise ValueError(f"Unknown method: {method}")

    return wave_dir_y, wave_dir_x


def compute_optical_flow(
    images: np.ndarray,
    method: str = "farneback",
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算光流场

    Args:
        images: (T, H, W) 图像序列
        method: 方法，'farneback' 或 'tv-l1'
        pyr_scale: 金字塔尺度
        levels: 金字塔层数
        winsize: 窗口大小

    Returns:
        flow_y: (T-1, H, W) y 方向光流
        flow_x: (T-1, H, W) x 方向光流
    """
    import cv2

    T, H, W = images.shape

    # 转换为 uint8
    if images.dtype != np.uint8:
        images_norm = []
        for t in range(T):
            frame = images[t]
            if frame.max() > frame.min():
                frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
            else:
                frame_norm = frame - frame.min()
            images_norm.append((frame_norm * 255).astype(np.uint8))
        images = np.stack(images_norm)

    flow_y = []
    flow_x = []

    for t in range(T - 1):
        prev = images[t]
        curr = images[t + 1]

        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr,
                None,
                pyr_scale=pyr_scale,
                levels=levels,
                winsize=winsize,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        flow_y.append(flow[..., 0])
        flow_x.append(flow[..., 1])

    return np.array(flow_y), np.array(flow_x)


def compute_angle_correlation(
    flow_y: np.ndarray,
    flow_x: np.ndarray,
    grad_y: np.ndarray,
    grad_x: np.ndarray
) -> Dict:
    """
    计算角度分布统计

    Args:
        flow_y: (T, H, W) y 方向光流
        flow_x: (T, H, W) x 方向光流
        grad_y: (T, H, W) y 方向梯度
        grad_x: (T, H, W) x 方向梯度

    Returns:
        角度统计字典
    """
    # 计算角度
    flow_angle = np.arctan2(flow_y, flow_x)
    grad_angle = np.arctan2(grad_y, grad_x)

    # 角度差
    angle_diff = np.abs(flow_angle - grad_angle)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # [0, π]

    # 对齐度（cos of angle difference）
    alignment = np.cos(angle_diff)

    return {
        "mean_angle_diff_rad": float(np.mean(angle_diff)),
        "mean_alignment": float(np.mean(alignment)),
        "alignment_per_frame": [float(np.mean(alignment[t])) for t in range(alignment.shape[0])],
    }
