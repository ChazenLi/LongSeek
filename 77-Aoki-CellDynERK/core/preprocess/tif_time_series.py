"""
TIF时序数据处理模块

用于处理生物显微镜TIF格式的时序数据，包括：
- 时间序列切割和提取
- 荧光信号数据提取
- 细胞迁移追踪
- 细胞变化分析
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image, ImageSequence
import tifffile
import json

logger = logging.getLogger(__name__)


class TIFTimeSeriesLoader:
    """
    TIF时序数据加载器
    
    支持两种格式：
    1. 单个多帧TIF（stack）
    2. 多个单帧TIF（文件序列）
    """
    
    def __init__(self, tif_path: str):
        """
        初始化TIF加载器
        
        Args:
            tif_path: TIF文件或目录路径
        """
        self.tif_path = Path(tif_path)
        
        # 判断是文件还是目录
        if self.tif_path.is_file():
            self.is_stack = True
            self.files = [self.tif_path]
        elif self.tif_path.is_dir():
            self.is_stack = False
            # 获取所有TIF文件并排序
            self.files = sorted(self.tif_path.glob("*.tif")) + sorted(self.tif_path.glob("*.tiff"))
        else:
            raise ValueError(f"Invalid TIF path: {tif_path}")
    
    def load_time_series(self) -> Dict[str, any]:
        """
        加载时间序列数据
        
        Returns:
            {
                'time_series': np.ndarray,  # (T, H, W, C) or (T, H, W)
                'time_points': List[float],  # 时间点
                'n_frames': int,  # 总帧数
                'frame_rate': float,  # 帧率
                'shape': Tuple[int, int, int, int],  # (T, H, W, C)
                'channels': Dict[str, np.ndarray],  # 各通道数据
            }
        """
        logger.info(f"Loading TIF time series from {self.tif_path}")
        
        if self.is_stack:
            # 单个多帧TIF
            with Image.open(self.tif_path) as img:
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    # 多帧图像序列
                    n_frames = img.n_frames
                    time_series = []
                    for i in range(n_frames):
                        img.seek(i)
                        frame = np.array(img)
                        time_series.append(frame)
                    
                    time_series = np.stack(time_series)
                    logger.info(f"Loaded {n_frames} frames from stack TIF")
                else:
                    # 单帧图像
                    time_series = np.array(img)[np.newaxis, ...]
                    n_frames = 1
                    logger.info("Loaded single frame TIF")
            
            return self._process_time_series(time_series)
        
        else:
            # 多个单帧TIF文件
            time_series = []
            for i, file_path in enumerate(self.files):
                with Image.open(file_path) as img:
                    frame = np.array(img)
                    time_series.append(frame)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Loaded {i+1}/{len(self.files)} frames")
            
            time_series = np.stack(time_series)
            logger.info(f"Loaded {len(self.files)} frames from TIF files")
            
            return self._process_time_series(time_series)
    
    def _process_time_series(self, time_series: np.ndarray) -> Dict[str, any]:
        """
        处理时间序列数据
        
        Args:
            time_series: (T, H, W) or (T, H, W, C) array
            
        Returns:
            Processed data dictionary
        """
        T, H, W = time_series.shape[:3]
        
        # 处理多通道
        if time_series.ndim == 4:
            C = time_series.shape[3]
            logger.info(f"Time series shape: ({T}, {H}, {W}, {C}) - {C} channels")
            
            # 分离各通道
            channels = {}
            channel_names = self._guess_channel_names(C)
            
            for c in range(C):
                channels[channel_names[c]] = time_series[:, :, :, c]
            
            # 如果是RGB，提取亮度通道作为灰度
            if C == 3:
                channels['gray'] = np.dot(time_series[:, :, :, :3], [0.299, 0.587, 0.114])
            
            # 默认使用第一个通道作为主数据
            main_data = time_series[:, :, :, 0]
        else:
            C = 1
            logger.info(f"Time series shape: ({T}, {H}, {W}) - single channel")
            channels = {'channel_0': time_series}
            channel_names = ['channel_0']
            main_data = time_series
        
        # 计算时间点（假设均匀采样）
        time_points = np.arange(T, dtype=float)
        frame_rate = 1.0  # 默认帧率，需要用户提供
        
        return {
            'time_series': time_series,
            'main_data': main_data,
            'time_points': time_points,
            'n_frames': T,
            'frame_rate': frame_rate,
            'shape': time_series.shape,
            'channels': channels,
            'channel_names': channel_names,
            'height': H,
            'width': W,
            'n_channels': C
        }
    
    def _guess_channel_names(self, n_channels: int) -> List[str]:
        """
        猜测通道名称
        
        Args:
            n_channels: 通道数
            
        Returns:
            通道名称列表
        """
        if n_channels == 1:
            return ['channel_0']
        elif n_channels == 2:
            return ['DAPI', 'GFP']  # 常见的2通道标记
        elif n_channels == 3:
            return ['DAPI', 'FITC', 'TRITC']  # 常见的3通道免疫荧光
        elif n_channels == 4:
            return ['DAPI', 'FITC', 'TRITC', 'CY5']  # 常见的4通道免疫荧光
        else:
            return [f'channel_{i}' for i in range(n_channels)]
    
    def split_by_frame_rate(self, frame_rate: float, time_points: Optional[List[float]] = None) -> Dict[str, any]:
        """
        按帧率切割时间序列
        
        Args:
            frame_rate: 帧率（帧/秒）
            time_points: 可选的时间点列表（秒）
            
        Returns:
            切割后的时间序列数据
        """
        data = self.load_time_series()
        time_series = data['time_series']
        
        T = data['n_frames']
        total_time = T / frame_rate
        
        logger.info(f"Total duration: {total_time:.2f} seconds ({T} frames @ {frame_rate} fps)")
        
        if time_points is None:
            # 默认：每10秒采样一次
            interval = 10.0  # 秒
            n_samples = int(total_time / interval)
            time_points = np.linspace(0, total_time, n_samples)
        
        # 根据时间点提取对应帧
        frames_at_time = []
        for t in time_points:
            frame_idx = int(t * frame_rate)
            frame_idx = min(frame_idx, T - 1)
            frames_at_time.append(time_series[frame_idx])
        
        frames_at_time = np.stack(frames_at_time)
        
        return {
            'time_points': time_points,
            'frames_at_time': frames_at_time,
            'frame_indices': [int(t * frame_rate) for t in time_points],
            'n_samples': len(time_points),
            'original_data': data
        }


class FluorescenceSignalExtractor:
    """
    荧光信号数据提取器
    
    用于从时序TIF中提取荧光信号强度、空间分布等信息
    """
    
    def __init__(self, tif_loader: TIFTimeSeriesLoader):
        self.tif_loader = tif_loader
    
    def extract_channel_signal(self, channel_name: str = 'channel_0') -> Dict[str, np.ndarray]:
        """
        提取特定通道的信号
        
        Args:
            channel_name: 通道名称
            
        Returns:
            {
                'signal_timeseries': np.ndarray,  # 每帧的总信号
                'signal_spatial': np.ndarray,  # 空间分布
                'signal_mean': np.ndarray,  # 每帧的平均信号
                'signal_std': np.ndarray,  # 每帧的标准差
                'signal_max': np.ndarray,  # 每帧的最大值
            }
        """
        data = self.tif_loader.load_time_series()
        channels = data['channels']
        
        if channel_name not in channels:
            raise ValueError(f"Channel {channel_name} not found. Available: {list(channels.keys())}")
        
        channel_data = channels[channel_name]  # (T, H, W)
        T, H, W = channel_data.shape
        
        # 计算时间序列信号
        signal_timeseries = np.mean(channel_data, axis=(1, 2))  # 每帧的全局平均
        
        # 计算空间分布（最后一帧）
        signal_spatial = channel_data[-1]
        
        # 计算每帧的统计
        signal_mean = np.mean(channel_data, axis=(1, 2))
        signal_std = np.std(channel_data, axis=(1, 2))
        signal_max = np.max(channel_data, axis=(1, 2))
        
        return {
            'channel_name': channel_name,
            'signal_timeseries': signal_timeseries,
            'signal_spatial': signal_spatial,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'signal_max': signal_max,
            'n_frames': T,
            'shape': channel_data.shape
        }
    
    def extract_all_channels(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        提取所有通道的信号
        
        Returns:
            字典：{channel_name: signal_data}
        """
        data = self.tif_loader.load_time_series()
        channel_names = data['channel_names']
        
        all_signals = {}
        for channel_name in channel_names:
            all_signals[channel_name] = self.extract_channel_signal(channel_name)
        
        return all_signals
    
    def extract_fluorescence_ratio(self, channel_a: str, channel_b: str) -> np.ndarray:
        """
        计算两个通道的荧光强度比
        
        Args:
            channel_a: 分子通道名
            channel_b: 分母通道名
            
        Returns:
            每帧的荧光比
        """
        signals = self.extract_all_channels()
        
        if channel_a not in signals:
            raise ValueError(f"Channel {channel_a} not found")
        if channel_b not in signals:
            raise ValueError(f"Channel {channel_b} not found")
        
        signal_a = signals[channel_a]['signal_mean']
        signal_b = signals[channel_b]['signal_mean']
        
        # 避免除以零
        ratio = np.divide(signal_a, signal_b, out=np.zeros_like(signal_a), where=signal_b!=0)
        
        return ratio
    
    def extract_temporal_dynamics(self, channel_name: str) -> Dict[str, np.ndarray]:
        """
        提取荧光信号的时序动力学特征
        
        Args:
            channel_name: 通道名称
            
        Returns:
            {
                'delta_F': np.ndarray,  # 帧间变化
                'normalized_F': np.ndarray,  # 归一化信号
                'activation_time': float,  # 激活时间
                'peak_time': float,  # 峰值时间
                'half_max_time': float,  # 半峰时间
            }
        """
        signals = self.extract_channel_signal(channel_name)
        F = signals['signal_timeseries']  # (T,)
        time_points = self.tif_loader.load_time_series()['time_points']
        
        # 归一化到[0,1]
        F_norm = (F - F.min()) / (F.max() - F.min() + 1e-10)
        
        # 计算帧间变化
        delta_F = np.diff(F)
        
        # 计算激活时间（首次超过50%归一化值的时间）
        half_max_idx = np.where(F_norm > 0.5)[0]
        if len(half_max_idx) > 0:
            half_max_time = time_points[half_max_idx[0]]
        else:
            half_max_time = 0.0
        
        # 计算峰值时间
        peak_idx = np.argmax(F)
        peak_time = time_points[peak_idx]
        
        return {
            'channel_name': channel_name,
            'F': F,
            'F_norm': F_norm,
            'delta_F': delta_F,
            'activation_time': half_max_time,
            'peak_time': peak_time,
            'half_max_time': half_max_time,
            'time_points': time_points
        }


class CellMigrationAnalyzer:
    """
    细胞迁移分析器
    
    用于从时序TIF中追踪和分析细胞迁移
    """
    
    def __init__(self, tif_loader: TIFTimeSeriesLoader):
        self.tif_loader = tif_loader
    
    def extract_wound_area_series(self, 
                               threshold: float = 127,
                               background: str = 'auto') -> Dict[str, np.ndarray]:
        """
        提取伤口面积时间序列
        
        Args:
            threshold: 二值化阈值
            background: 背景类型 ('auto'=亮背景, 'dark'=暗背景)
            
        Returns:
            {
                'wound_area': np.ndarray,  # 伤口面积（像素）
                'normalized_area': np.ndarray,  # 归一化面积
                'wound_center': np.ndarray,  # 伤口中心
                'cell_coverage': np.ndarray,  # 细胞覆盖率
            }
        """
        data = self.tif_loader.load_time_series()
        main_data = data['main_data']  # (T, H, W)
        time_points = data['time_points']
        
        T, H, W = main_data.shape
        
        # 二值化
        if background == 'auto':
            # 自动判断背景（假设伤口区较暗或较亮）
            mean_val = np.mean(main_data[0])
            background = 'dark' if mean_val > 127 else 'light'
        
        if background == 'dark':
            binary = (main_data < threshold).astype(np.uint8)
            wound_pixels = 1  # 暗背景是伤口
            cell_pixels = 0
        else:
            binary = (main_data > threshold).astype(np.uint8)
            wound_pixels = 0  # 亮背景，伤口是较暗区域
            cell_pixels = 1
        
        # 计算每帧的伤口面积
        wound_area = []
        cell_coverage = []
        wound_center = []
        
        for t in range(T):
            frame = binary[t]
            area = np.sum(frame == wound_pixels)
            wound_area.append(float(area))
            cell_coverage.append(np.sum(frame == cell_pixels) / (H * W))
            
            # 计算伤口中心
            rows, cols = np.where(frame == wound_pixels)
            if len(rows) > 0:
                center_y = np.mean(rows)
                center_x = np.mean(cols)
                wound_center.append([center_y, center_x])
            else:
                wound_center.append([H/2, W/2])
        
        wound_area = np.array(wound_area)
        cell_coverage = np.array(cell_coverage)
        wound_center = np.array(wound_center)
        
        # 归一化
        normalized_area = wound_area / wound_area[0]
        
        return {
            'wound_area': wound_area,
            'normalized_area': normalized_area,
            'wound_center': wound_center,
            'cell_coverage': cell_coverage,
            'time_points': time_points,
            'threshold': threshold,
            'background': background
        }
    
    def extract_cell_migration_velocity(self, 
                                    threshold: float = 127,
                                    background: str = 'auto') -> Dict[str, any]:
        """
        估算细胞迁移速度
        
        Args:
            threshold: 二值化阈值
            background: 背景类型
            
        Returns:
            {
                'velocity_x': np.ndarray,  # X方向速度
                'velocity_y': np.ndarray,  # Y方向速度
                'velocity_magnitude': np.ndarray,  # 速度大小
                'velocity_angle': np.ndarray,  # 速度角度
            }
        """
        data = self.tif_loader.load_time_series()
        wound_data = self.extract_wound_area_series(threshold, background)
        
        wound_center = wound_data['wound_center']  # (T, 2)
        time_points = data['time_points']  # (T,)
        
        # 计算伤口中心移动速度
        center_displacement = np.diff(wound_center, axis=0)  # (T-1, 2)
        time_interval = np.diff(time_points)  # (T-1,)
        
        # 速度 = 位移 / 时间
        velocity = center_displacement / time_interval[:, np.newaxis]  # (T-1, 2)
        
        # 计算速度大小和角度
        velocity_magnitude = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
        velocity_angle = np.arctan2(velocity[:, 1], velocity[:, 0])  # Y, X
        
        return {
            'velocity_x': velocity[:, 0],
            'velocity_y': velocity[:, 1],
            'velocity_magnitude': velocity_magnitude,
            'velocity_angle': velocity_angle,
            'wound_center': wound_center,
            'time_intervals': time_interval,
            'velocity': velocity
        }
    
    def analyze_morphological_changes(self, 
                                   threshold: float = 127,
                                   background: str = 'auto') -> Dict[str, any]:
        """
        分析形态变化
        
        Args:
            threshold: 二值化阈值
            background: 背景类型
            
        Returns:
            {
                'aspect_ratio': np.ndarray,  # 长宽比
                'circularity': np.ndarray,  # 圆度
                'solidity': np.ndarray,  # 凸性
                'perimeter': np.ndarray,  # 周长
            }
        """
        from skimage.measure import regionprops, label
        from skimage.morphology import convex_hull_image
        import cv2
        
        data = self.tif_loader.load_time_series()
        main_data = data['main_data']  # (T, H, W)
        
        if background == 'dark':
            binary = (main_data < threshold).astype(np.uint8)
        else:
            binary = (main_data > threshold).astype(np.uint8)
        
        T, H, W = main_data.shape
        
        aspect_ratio = []
        circularity = []
        solidity = []
        perimeter = []
        
        for t in range(T):
            frame = binary[t]
            
            # 标记连通区域
            labeled = label(frame)
            regions = regionprops(labeled)
            
            if len(regions) > 0:
                # 使用最大的区域（伤口）
                largest_region = max(regions, key=lambda r: r.area)
                
                aspect_ratio.append(largest_region.major_axis_length / (largest_region.minor_axis_length + 1e-10))
                
                # 圆度 = (4 * pi * area) / perimeter^2
                perimeter = largest_region.perimeter
                area = largest_region.area
                if perimeter > 0:
                    circularity.append(4 * np.pi * area / (perimeter ** 2 + 1e-10))
                else:
                    circularity.append(0.0)
                
                # 凸性 = area / convex_hull_area
                hull = convex_hull_image(frame)
                hull_area = np.sum(hull)
                if hull_area > 0:
                    solidity.append(area / hull_area)
                else:
                    solidity.append(1.0)
                
                perimeter.append(perimeter)
            else:
                aspect_ratio.append(1.0)
                circularity.append(0.0)
                solidity.append(1.0)
                perimeter.append(0.0)
        
        return {
            'aspect_ratio': np.array(aspect_ratio),
            'circularity': np.array(circularity),
            'solidity': np.array(solidity),
            'perimeter': np.array(perimeter),
            'time_points': data['time_points']
        }


class TIFDataExporter:
    """
    TIF数据导出器
    
    将提取的数据导出为JSON、NPY等格式
    """
    
    @staticmethod
    def export_to_json(data: Dict, output_path: str):
        """
        导出为JSON格式
        
        Args:
            data: 数据字典
            output_path: 输出路径
        """
        # 将numpy数组转换为列表
        export_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                export_data[key] = {
                    'array': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, dict):
                export_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        export_data[key][sub_key] = {
                            'array': sub_value.tolist(),
                            'shape': sub_value.shape,
                            'dtype': str(sub_value.dtype)
                        }
                    else:
                        export_data[key][sub_key] = sub_value
            else:
                export_data[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported data to {output_path}")
    
    @staticmethod
    def export_to_npy(data: Dict, output_path: str):
        """
        导出为NPY格式
        
        Args:
            data: 数据字典
            output_path: 输出路径
        """
        np.savez_compressed(output_path, **data)
        logger.info(f"Exported data to {output_path}")
    
    @staticmethod
    def export_time_series(time_series: np.ndarray, 
                            time_points: np.ndarray,
                            output_dir: str,
                            prefix: str = 'frame'):
        """
        导出时间序列为单独的图像
        
        Args:
            time_series: (T, H, W) or (T, H, W, C) array
            time_points: 时间点
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        T = time_series.shape[0]
        
        for t in range(T):
            if time_series.ndim == 4:
                # 多通道
                frame_data = time_series[t]
            else:
                frame_data = time_series[t]
            
            # 转换为uint8
            if frame_data.dtype != np.uint8:
                frame_data = ((frame_data - frame_data.min()) / 
                           (frame_data.max() - frame_data.min()) * 255).astype(np.uint8)
            
            # 保存
            output_path = output_dir / f"{prefix}_t{t:04d}_time{time_points[t]:.1f}.png"
            Image.fromarray(frame_data).save(output_path)
            
            if (t + 1) % 10 == 0:
                logger.info(f"Exported {t+1}/{T} frames")
        
        logger.info(f"Exported all frames to {output_dir}")


def process_tif_dataset(tif_path: str,
                        frame_rate: float = 1.0,
                        threshold: float = 127,
                        background: str = 'auto',
                        output_dir: str = 'output') -> Dict[str, any]:
    """
    完整处理TIF数据集
    
    Args:
        tif_path: TIF文件或目录路径
        frame_rate: 帧率（帧/秒）
        threshold: 二值化阈值
        background: 背景类型
        output_dir: 输出目录
        
    Returns:
        完整的处理结果
    """
    logger.info("="*70)
    logger.info("Processing TIF Time Series Dataset")
    logger.info("="*70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载TIF时间序列
    tif_loader = TIFTimeSeriesLoader(tif_path)
    ts_data = tif_loader.load_time_series()
    
    logger.info(f"\nStep 1: Loaded TIF data")
    logger.info(f"  Shape: {ts_data['shape']}")
    logger.info(f"  N frames: {ts_data['n_frames']}")
    logger.info(f"  Channels: {ts_data['channel_names']}")
    
    # 2. 荧光信号提取
    logger.info(f"\nStep 2: Extracting fluorescence signals")
    fluo_extractor = FluorescenceSignalExtractor(tif_loader)
    fluo_signals = fluo_extractor.extract_all_channels()
    
    for channel_name, signal_data in fluo_signals.items():
        logger.info(f"  Channel {channel_name}:")
        logger.info(f"    Mean signal range: [{signal_data['signal_mean'].min():.2f}, {signal_data['signal_mean'].max():.2f}]")
        logger.info(f"    Std range: [{signal_data['signal_std'].min():.2f}, {signal_data['signal_std'].max():.2f}]")
    
    # 3. 细胞迁移分析
    logger.info(f"\nStep 3: Analyzing cell migration")
    migration_analyzer = CellMigrationAnalyzer(tif_loader)
    wound_area = migration_analyzer.extract_wound_area_series(threshold, background)
    velocity = migration_analyzer.extract_cell_migration_velocity(threshold, background)
    morphology = migration_analyzer.analyze_morphological_changes(threshold, background)
    
    logger.info(f"  Initial wound area: {wound_area['wound_area'][0]:.0f} pixels")
    logger.info(f"  Final wound area: {wound_area['wound_area'][-1]:.0f} pixels")
    logger.info(f"  Closure rate: {(1 - wound_area['wound_area'][-1]/wound_area['wound_area'][0])*100:.1f}%")
    logger.info(f"  Mean velocity: {np.mean(velocity['velocity_magnitude']):.2f} pixels/frame")
    
    # 4. 按帧率切割（如果需要）
    if frame_rate is not None and frame_rate != 1.0:
        logger.info(f"\nStep 4: Splitting by frame rate ({frame_rate} fps)")
        split_data = tif_loader.split_by_frame_rate(frame_rate)
        logger.info(f"  Split into {split_data['n_samples']} time points")
    else:
        split_data = None
    
    # 5. 导出数据
    logger.info(f"\nStep 5: Exporting data")
    
    # 导出为JSON
    TIFDataExporter.export_to_json({
        'time_series_data': ts_data,
        'fluorescence_signals': fluo_signals,
        'wound_area_series': wound_area,
        'migration_velocity': velocity,
        'morphological_changes': morphology,
        'split_data': split_data,
        'processing_params': {
            'tif_path': tif_path,
            'frame_rate': frame_rate,
            'threshold': threshold,
            'background': background
        }
    }, output_path / 'complete_results.json')
    
    # 导出为NPY
    TIFDataExporter.export_to_npy({
        'time_series': ts_data['time_series'],
        'main_data': ts_data['main_data'],
        'wound_area': wound_area['wound_area'],
        'normalized_area': wound_area['normalized_area'],
        'velocity': velocity['velocity'],
        'morphology': morphology
    }, output_path / 'data.npz')
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"Output directory: {output_path}")
    logger.info("="*70)
    
    return {
        'time_series_data': ts_data,
        'fluorescence_signals': fluo_signals,
        'wound_area_series': wound_area,
        'migration_velocity': velocity,
        'morphological_changes': morphology,
        'split_data': split_data,
        'output_dir': str(output_path)
    }


if __name__ == "__main__":
    import sys
    
    # 测试TIF加载
    if len(sys.argv) > 1:
        tif_path = sys.argv[1]
    else:
        # 使用示例数据
        tif_path = "path/to/your/tif_file.tif"
        print("Usage: python tif_time_series.py <tif_path> [frame_rate] [threshold]")
        print("\nCreating test data for demonstration...")
        
        # 创建示例数据
        t, h, w = 30, 256, 256
        test_data = np.random.randint(0, 255, (t, h, w), dtype=np.uint8)
        
        # 添加伤口（中心区域）
        center_y, center_x = h//2, w//2
        radius = 40
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
        
        for i in range(t):
            test_data[i, mask] = 50  # 伤口区域设为暗（低信号）
        
        # 保存测试数据
        Path("test_output").mkdir(exist_ok=True)
        from tifffile import TiffWriter
        with TiffWriter("test_output/test_time_series.tif") as tif:
            for i in range(t):
                tif.write(test_data[i], metadata={'frame': str(i)})
        
        tif_path = "test_output/test_time_series.tif"
        print(f"Created test TIF: {tif_path}")
    
    # 处理数据
    frame_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 127.0
    
    result = process_tif_dataset(
        tif_path=tif_path,
        frame_rate=frame_rate,
        threshold=threshold,
        output_dir='output'
    )
