"""
分子场处理模块
ERK/MLC场的预处理、梯度计算和耦合分析
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MolecularFieldProcessor:
    """分子场处理器"""
    
    def __init__(self, 
                 field_type: str = 'ERK',
                 pixel_size: float = 0.49,
                 dt: Optional[float] = None):
        """
        Args:
            field_type: 场类型 ('ERK', 'MLC', 'ratio')
            pixel_size: 像素大小 (微米)
            dt: 时间间隔 (秒)
        """
        self.field_type = field_type
        self.pixel_size = pixel_size
        self.dt = dt
    
    def preprocess_field(self, 
                        field: np.ndarray,
                        remove_nan: bool = True,
                        correct_drift: bool = True,
                        correct_bleaching: bool = True,
                        smooth_sigma: float = 1.0) -> np.ndarray:
        """
        ERK场预处理
        
        Args:
            field: 原始场 (T, H, W) or (H, W)
            remove_nan: 是否移除NaN/Inf
            correct_drift: 是否校正全局漂移
            correct_bleaching: 是否校正photobleaching
            smooth_sigma: 高斯平滑参数
            
        Returns:
            预处理后的场
        """
        field = field.copy()
        
        # 检查输入
        if field.ndim == 2:
            field = field[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # 1. NaN/Inf检测与插值
        if remove_nan:
            field = self._remove_nan_inf(field)
        
        # 2. 全局漂移校正
        if correct_drift and field.shape[0] > 1:
            field = self._correct_drift(field)
        
        # 3. Photobleaching校正
        if correct_bleaching and field.shape[0] > 1:
            field = self._correct_bleaching(field)
        
        # 4. 空间滤波
        if smooth_sigma > 0:
            field = self._spatial_filter(field, sigma=smooth_sigma)
        
        return field
    
    def _remove_nan_inf(self, field: np.ndarray) -> np.ndarray:
        """移除NaN/Inf并插值"""
        # 检测NaN
        nan_mask = np.isnan(field)
        
        # 检测Inf
        inf_mask = np.isinf(field)
        
        if nan_mask.any() or inf_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN and {inf_mask.sum()} Inf in field")
            
            # 用中位数替换
            field_clean = field.copy()
            field_clean[nan_mask | inf_mask] = np.nanmedian(field[~(nan_mask | inf_mask)])
            
            # 邻域平滑
            field_clean = ndimage.median_filter(field_clean, size=3)
            
            return field_clean
        
        return field
    
    def _correct_drift(self, field: np.ndarray) -> np.ndarray:
        """全局漂移校正（使用phase correlation）"""
        T, H, W = field.shape
        corrected = field.copy()
        
        # 第一帧作为参考
        reference = corrected[0]
        
        for t in range(1, T):
            # 计算位移（使用phase correlation）
            shift, error, diffphase = self._phase_correlation(reference, corrected[t])
            
            # 应用位移校正
            if shift[0] != 0 or shift[1] != 0:
                corrected[t] = ndimage.shift(corrected[t], shift, mode='nearest')
        
        return corrected
    
    def _phase_correlation(self, 
                          ref: np.ndarray, 
                          target: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
        """Phase correlation计算位移"""
        from scipy.signal import correlate2d
        
        # FFT
        f1 = np.fft.fft2(ref)
        f2 = np.fft.fft2(target)
        
        # Cross-power spectrum
        product = f1 * np.conj(f2)
        cross_power_spectrum = product / (np.abs(product) + 1e-10)
        
        # IFFT
        correlation = np.fft.ifft2(cross_power_spectrum).real
        
        # 找到峰值
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # 转换为位移（需要考虑wrap-around）
        y_shift = peak[0] if peak[0] < ref.shape[0] // 2 else peak[0] - ref.shape[0]
        x_shift = peak[1] if peak[1] < ref.shape[1] // 2 else peak[1] - ref.shape[1]
        
        return (y_shift, x_shift), 0.0, 0.0
    
    def _correct_bleaching(self, field: np.ndarray) -> np.ndarray:
        """Photobleaching校正（每帧robust normalization）"""
        T, H, W = field.shape
        corrected = field.copy()
        
        # 计算每帧的中位数强度
        frame_medians = np.median(corrected, axis=(1, 2))
        
        # 使用第一帧作为参考
        reference_median = frame_medians[0]
        
        # 校正每帧
        for t in range(T):
            if frame_medians[t] > 0:
                scale_factor = reference_median / frame_medians[t]
                corrected[t] *= scale_factor
        
        return corrected
    
    def _spatial_filter(self, 
                       field: np.ndarray, 
                       sigma: float = 1.0) -> np.ndarray:
        """空间滤波"""
        filtered = np.zeros_like(field)
        
        for t in range(field.shape[0]):
            filtered[t] = gaussian_filter(field[t], sigma=sigma)
        
        return filtered
    
    def compute_gradient(self, 
                        field: np.ndarray,
                        sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算ERK场梯度
        
        Args:
            field: ERK场 (T, H, W) or (H, W)
            sigma: 梯度计算前的高斯平滑参数
            
        Returns:
            grad_y: y方向梯度 (T, H, W) or (H, W)
            grad_x: x方向梯度 (T, H, W) or (H, W)
            grad_magnitude: 梯度幅值
        """
        # 如果是时间序列，先平滑
        if field.ndim == 3 and sigma > 0:
            field_smooth = self._spatial_filter(field, sigma=sigma)
        elif field.ndim == 2 and sigma > 0:
            field_smooth = gaussian_filter(field, sigma=sigma)
        else:
            field_smooth = field
        
        # 计算梯度
        grad_y, grad_x = np.gradient(field_smooth, axis=(-2, -1))
        
        # 计算幅值
        grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        return grad_y, grad_x, grad_magnitude
    
    def compute_gradient_direction(self, 
                                  grad_y: np.ndarray,
                                  grad_x: np.ndarray) -> np.ndarray:
        """
        计算梯度方向（角度）
        
        Returns:
            angle: 方向角（弧度），范围[-pi, pi]
        """
        angle = np.arctan2(grad_y, grad_x)
        return angle
    
    def extract_wave_properties(self, 
                                field: np.ndarray,
                                threshold: float = 0.5) -> Dict:
        """
        提取波的性质
        
        Args:
            field: ERK场 (T, H, W)
            threshold: 波识别阈值
            
        Returns:
            Dict包含波速、波前方向等
        """
        if field.ndim != 3:
            raise ValueError("Wave analysis requires time series (T, H, W)")
        
        T, H, W = field.shape
        
        # 计算梯度
        grad_y, grad_x, grad_mag = self.compute_gradient(field)
        
        # 识别波前（高梯度区域）
        is_wavefront = grad_mag > threshold * np.max(grad_mag, axis=(1, 2), keepdims=True)
        
        # 估计波速（基于波前位移）
        wave_speeds = self._estimate_wave_speed(field, is_wavefront)
        
        # 估计波前方向（基于梯度方向）
        wave_directions = self._estimate_wave_direction(grad_y, grad_x, is_wavefront)
        
        return {
            'wave_speed': wave_speeds,
            'wave_direction': wave_directions,
            'wavefront_mask': is_wavefront,
            'gradient_magnitude': grad_mag
        }
    
    def _estimate_wave_speed(self, 
                            field: np.ndarray,
                            wavefront_mask: np.ndarray) -> List[float]:
        """估计波速"""
        T = field.shape[0]
        speeds = []
        
        for t in range(1, T):
            # 波前中心
            prev_centers = self._find_wavefront_center(wavefront_mask[t-1])
            curr_centers = self._find_wavefront_center(wavefront_mask[t])
            
            if prev_centers is not None and curr_centers is not None:
                # 计算位移
                dy = curr_centers[0] - prev_centers[0]
                dx = curr_centers[1] - prev_centers[1]
                distance = np.sqrt(dy**2 + dx**2)
                
                # 转换为实际距离（微米）
                distance_um = distance * self.pixel_size
                
                # 转换为时间（秒）
                if self.dt is not None:
                    time_s = self.dt
                else:
                    time_s = 1.0
                
                speed = distance_um / time_s
                speeds.append(speed)
        
        return speeds
    
    def _find_wavefront_center(self, wavefront_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """找到波前中心"""
        if not wavefront_mask.any():
            return None
        
        y_indices, x_indices = np.where(wavefront_mask)
        center_y = float(np.mean(y_indices))
        center_x = float(np.mean(x_indices))
        
        return (center_y, center_x)
    
    def _estimate_wave_direction(self, 
                               grad_y: np.ndarray,
                               grad_x: np.ndarray,
                               wavefront_mask: np.ndarray) -> List[float]:
        """估计波前方向"""
        T = wavefront_mask.shape[0]
        directions = []
        
        for t in range(T):
            if not wavefront_mask[t].any():
                continue
            
            # 计算平均梯度方向
            masked_grad_y = grad_y[t][wavefront_mask[t]]
            masked_grad_x = grad_x[t][wavefront_mask[t]]
            
            avg_grad_y = np.mean(masked_grad_y)
            avg_grad_x = np.mean(masked_grad_x)
            
            # 计算角度
            angle = np.arctan2(avg_grad_y, avg_grad_x)
            directions.append(angle)
        
        return directions


def compute_angle_between_vectors(v1: np.ndarray,
                                v2: np.ndarray,
                                normalize: bool = True) -> np.ndarray:
    """
    计算两个向量场之间的角度
    
    Args:
        v1: 第一个向量场 (..., 2)
        v2: 第二个向量场 (..., 2)
        normalize: 是否归一化向量
        
    Returns:
        angle: 角度（弧度），范围[0, pi]
    """
    if normalize:
        norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-10
        norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-10
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
    else:
        v1_norm = v1
        v2_norm = v2
    
    # 点积
    dot_product = np.sum(v1_norm * v2_norm, axis=-1)
    
    # 截断到[-1, 1]以避免数值误差
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算角度
    angle = np.arccos(dot_product)
    
    return angle


def compute_coupling_metrics(mask: np.ndarray,
                            field: np.ndarray,
                            velocity_field: Optional[np.ndarray] = None,
                            pixel_size: float = 0.49) -> Dict:
    """
    计算分子-力学耦合指标

    Args:
        mask: 细胞掩膜 (H, W)
        field: 分子场 (H, W) or (T, H, W)
        velocity_field: 速度场 (H, W, 2) - 可选
        pixel_size: 像素大小

    Returns:
        Dict包含耦合指标
    """
    processor = MolecularFieldProcessor(pixel_size=pixel_size)

    # 确保field是2D的
    if field.ndim == 3:
        field_2d = field[-1]  # 使用最后一帧
    elif field.ndim == 2:
        field_2d = field
    else:
        logger.warning(f"Unexpected field shape: {field.shape}")
        field_2d = np.zeros_like(mask, dtype=np.float32)

    # 修复4：维度匹配 - 如果field和mask尺寸不同，resize field
    if field_2d.shape != mask.shape:
        from skimage.transform import resize
        field_2d = resize(field_2d, mask.shape,
                          preserve_range=True, anti_aliasing=True).astype(np.float32)
        logger.info(f"Field resized from {field.shape} to {field_2d.shape}")
    else:
        field_2d = field_2d.astype(np.float32)

    # 计算场梯度
    grad_y, grad_x, grad_mag = processor.compute_gradient(field_2d)

    results = {
        'gradient_magnitude': grad_mag,
        'gradient_y': grad_y,
        'gradient_x': grad_x,
        'gradient_direction': np.arctan2(grad_y, grad_x),
        'field_shape': field.shape,
        'field_used_shape': field_2d.shape,
    }

    # 如果有速度场，计算角度耦合
    if velocity_field is not None:
        # 检查维度匹配
        if velocity_field.shape[:2] == mask.shape:
            # 构造梯度向量场
            grad_field = np.stack([grad_y, grad_x], axis=-1)

            # 计算角度
            angle_v_grad = compute_angle_between_vectors(velocity_field, grad_field)

            results.update({
                'angle_v_grad_C': angle_v_grad,
                'mean_angle': np.mean(angle_v_grad),
                'std_angle': np.std(angle_v_grad),
                'angle_histogram': np.histogram(angle_v_grad, bins=36, range=(0, np.pi))
            })
        else:
            logger.warning(f"Velocity field shape {velocity_field.shape} doesn't match mask {mask.shape}")

    # 没有速度场，用迁移方向近似
    if 'angle_v_grad_C' not in results:
        # 估计从细胞区域到伤口的迁移方向
        wound_mask = (mask == 0).astype(np.float32)
        dist_to_wound = distance_transform_edt(wound_mask)

        # 伤口中心的梯度作为近似迁移方向
        mig_grad_y, mig_grad_x = np.gradient(dist_to_wound)
        mig_field = np.stack([mig_grad_y, mig_grad_x], axis=-1)

        # 计算角度
        angle_v_grad = compute_angle_between_vectors(mig_field,
                                                    np.stack([grad_y, grad_x], axis=-1))

        results.update({
            'angle_v_grad_C': angle_v_grad,
            'mean_angle': np.mean(angle_v_grad),
            'std_angle': np.std(angle_v_grad)
        })

    return results


if __name__ == "__main__":
    # 测试模块
    print("Testing Molecular Field Processor")
    
    # 创建合成ERK场
    H, W = 100, 100
    T = 10
    
    # 创建波动的ERK场
    field = np.zeros((T, H, W))
    for t in range(T):
        x = np.linspace(-3, 3, W)
        y = np.linspace(-3, 3, H)
        X, Y = np.meshgrid(x, y)
        
        # 波动 + 径向梯度
        wave = np.sin(X + t * 0.5) + 0.5 * (1 - np.sqrt(X**2 + Y**2))
        field[t] = wave
    
    # 创建processor
    processor = MolecularFieldProcessor(pixel_size=0.49, dt=180)
    
    # 预处理
    field_clean = processor.preprocess_field(field, smooth_sigma=1.0)
    print(f"Preprocessed field shape: {field_clean.shape}")
    
    # 计算梯度
    grad_y, grad_x, grad_mag = processor.compute_gradient(field_clean)
    print(f"Gradient shape: {grad_y.shape}")
    print(f"Gradient magnitude range: [{grad_mag.min():.3f}, {grad_mag.max():.3f}]")
    
    # 提取波性质
    wave_props = processor.extract_wave_properties(field_clean, threshold=0.5)
    print(f"Wave speeds: {wave_props['wave_speed']}")
    
    # 测试耦合指标
    mask = np.ones((H, W))
    mask[40:60, 40:60] = 0
    
    coupling = compute_coupling_metrics(mask, field_clean[0])
    print(f"Coupling metrics:")
    print(f"  Mean angle: {coupling['mean_angle']:.3f} rad")
    print(f"  Std angle: {coupling['std_angle']:.3f} rad")
    
    print("\nTest complete!")
