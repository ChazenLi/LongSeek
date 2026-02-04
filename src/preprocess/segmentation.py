"""
分割/掩膜模块 - Wound Mask 生成

参考 plan.md 第 5 节：
- 先做 wound mask，别一上来追 instance segmentation
- 最小可行：wound mask 自动生成
"""

import numpy as np
from skimage import filters, morphology, measure
from typing import Tuple, Optional


def generate_wound_mask_auto(
    image: np.ndarray,
    method: str = "sauvola",
    min_size: int = 1000,
    closing_size: int = 5,
    opening_size: int = 3
) -> np.ndarray:
    """
    自动生成 wound/empty 掩膜

    参考 plan.md 5.1:
    1. 局部阈值把低纹理空区分出来
    2. 形态学 closing + largest component
    3. 返回 0=wound/empty, 1=cell

    Args:
        image: (H, W) 单帧图像
        method: 阈值方法，'sauvola', 'phansalkar', 'otsu', 'adaptive'
        min_size: 最小连通区域大小
        closing_size: 形态学 closing 核大小
        opening_size: 形态学 opening 核大小

    Returns:
        mask: (H, W) 二值掩膜，1=cell, 0=wound/empty
    """
    # 归一化
    if image.max() > image.min():
        image_norm = (image - image.min()) / (image.max() - image.min())
    else:
        image_norm = image - image.min()

    image_uint8 = (image_norm * 255).astype(np.uint8)

    # 阈值分割
    if method == "sauvola":
        try:
            thresh = filters.threshold_sauvola(image_uint8, window_size=51)
        except:
            # 图像太小时回退到 Otsu
            thresh = filters.threshold_otsu(image_uint8)
    elif method == "phansalkar":
        try:
            thresh = filters.threshold_sauvola(
                image_uint8, window_size=51, k=0.3
            )  # Phansalkar 近似
        except:
            thresh = filters.threshold_otsu(image_uint8)
    elif method == "otsu":
        thresh = filters.threshold_otsu(image_uint8)
    elif method == "adaptive":
        thresh = filters.threshold_local(image_uint8, block_size=51, method='gaussian')
    else:
        raise ValueError(f"Unknown method: {method}")

    # 二值化：暗区域 = wound/empty（假设 wound 空区较暗）
    # 注意：实际数据可能相反，需要根据具体数据调整
    binary = image_uint8 > thresh

    # 形态学处理
    if closing_size > 0:
        binary = morphology.closing(
            binary,
            morphology.disk(closing_size)
        )

    if opening_size > 0:
        binary = morphology.opening(
            binary,
            morphology.disk(opening_size)
        )

    # 保留最大连通区域
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    if regions:
        areas = [r.area for r in regions]
        max_label = np.argmax(areas) + 1
        binary = labeled == max_label

    # 反转：确保 1=cell, 0=wound
    # 根据数据特性，可能需要调整这里的逻辑
    mask = binary.astype(np.uint8)

    # 后处理：确保边缘合理
    return mask


def generate_wound_mask_sequence(
    images: np.ndarray,
    method: str = "sauvola",
    temporal_smooth: bool = True,
    iou_threshold: float = 0.7
) -> np.ndarray:
    """
    为图像序列生成 wound mask

    Args:
        images: (T, H, W) 图像序列
        method: 阈值方法
        temporal_smooth: 是否进行时间平滑
        iou_threshold: IoU 检查阈值（用于 QC）

    Returns:
        masks: (T, H, W) 二值掩膜序列
    """
    T, H, W = images.shape
    masks = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        masks[t] = generate_wound_mask_auto(images[t], method=method)

    # 时间一致性检查
    if temporal_smooth and T > 1:
        masks = temporal_consistency_smooth(masks, iou_threshold=iou_threshold)

    return masks


def temporal_consistency_smooth(
    masks: np.ndarray,
    iou_threshold: float = 0.7,
    window_size: int = 3
) -> np.ndarray:
    """
    时间一致性平滑

    Args:
        masks: (T, H, W) 掩膜序列
        iou_threshold: IoO 阈值，低于此值则报警
        window_size: 平滑窗口大小

    Returns:
        平滑后的掩膜
    """
    T = masks.shape[0]
    smoothed = masks.copy()

    # 检查相邻帧的 IoU
    for t in range(1, T):
        iou = compute_iou(masks[t-1], masks[t])
        if iou < iou_threshold:
            # 可能有问题，进行平滑处理
            # 使用前一帧和当前帧的多数投票
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            window = masks[start:end]
            smoothed[t] = (window.sum(axis=0) > window_size // 2).astype(np.uint8)

    return smoothed


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算 IoU（Intersection over Union）

    Args:
        mask1: (H, W) 掩膜
        mask2: (H, W) 掩膜

    Returns:
        IoU 值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def refine_mask_boundary(
    mask: np.ndarray,
    erosion_size: int = 2
) -> np.ndarray:
    """
    细化掩膜边界（用于提取前沿区域）

    Args:
        mask: (H, W) 掩膜
        erosion_size: 腐蚀大小

    Returns:
        boundary: (H, W) 边界掩膜
    """
    if erosion_size > 0:
        eroded = morphology.erosion(mask, morphology.disk(erosion_size))
    else:
        eroded = mask

    boundary = mask.astype(np.int32) - eroded.astype(np.int32)
    return (boundary > 0).astype(np.uint8)


def extract_front_region(
    mask: np.ndarray,
    band_width: int = 10
) -> np.ndarray:
    """
    提取 wound 前沿区域（用于分析）

    Args:
        mask: (H, W) 掩膜
        band_width: 前沿带宽度（像素）

    Returns:
        front_region: (H, W) 前沿区域掩膜
    """
    # 距离变换
    from scipy import ndimage
    distance = ndimage.distance_transform_edt(1 - mask)

    # 前沿是 wound 区域中距离 cell 边界一定范围内的像素
    front_region = (distance > 0) & (distance <= band_width)
    return front_region.astype(np.uint8)


def compute_mask_statistics(mask: np.ndarray) -> dict:
    """
    计算掩膜统计信息

    Args:
        mask: (T, H, W) 掩膜序列

    Returns:
        统计信息字典
    """
    stats = {
        "area_per_frame": [],  # 每帧 cell 面积
        "wound_area_per_frame": [],  # 每帧 wound 面积
        "boundary_length_per_frame": [],  # 每帧边界长度
        "roughness_per_frame": [],  # 每帧粗糙度
    }

    for t in range(mask.shape[0]):
        m = mask[t]
        cell_area = m.sum()
        wound_area = (1 - m).sum()
        stats["area_per_frame"].append(int(cell_area))
        stats["wound_area_per_frame"].append(int(wound_area))

        # 边界长度（使用周长）
        if cell_area > 0:
            contours = measure.find_contours(m, 0.5)
            boundary_length = sum(len(c) for c in contours)
        else:
            boundary_length = 0
        stats["boundary_length_per_frame"].append(float(boundary_length))

        # 粗糙度（边界长度 / 凸包周长）
        if cell_area > 0 and contours:
            try:
                hull = morphology.convex_hull_image(m)
                hull_contours = measure.find_contours(hull, 0.5)
                hull_perimeter = max(len(c) for c in hull_contours) if hull_contours else 1
                roughness = boundary_length / hull_perimeter if hull_perimeter > 0 else 1
            except:
                roughness = 1.0
        else:
            roughness = 1.0
        stats["roughness_per_frame"].append(float(roughness))

    return stats
