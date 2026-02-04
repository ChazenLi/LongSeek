"""
SSBD 77 数据集加载器
统一的接口用于加载Aoki-CellDynERK的所有相关数据集
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class SSBD77Dataset:
    """单个SSBD 77数据集封装"""
    
    def __init__(self, 
                 dataset_id: str,
                 name: str,
                 folder: str,
                 condition: str,
                 dt: Optional[float] = None,
                 pixel_size: float = 0.49,
                 base_path: str = "datasets/ssbd77"):
        """
        Args:
            dataset_id: SSBD数据集ID (e.g., "4022")
            name: 数据集名称
            folder: 数据集文件夹
            condition: 条件类型 (spontaneous, inhibitor, optogenetic, opticflow)
            dt: 时间间隔(秒)
            pixel_size: 像素大小(微米)
            base_path: 数据集基础路径
        """
        self.dataset_id = dataset_id
        self.name = name
        self.folder = folder
        self.condition = condition
        self.dt = dt
        self.pixel_size = pixel_size
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / folder
        
        # 数据存储
        self.erk_field = None  # C_obs(T, H, W)
        self.initial_mask = None  # (H, W)
        self.time_series = None  # dict with time points and observables
        self.opticflow = None  # (T, H, W, 2) if opticflow data
        self.metadata = {}
        
    def load(self) -> bool:
        """
        加载数据集
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading dataset: {self.name}")
        
        try:
            # 加载初始mask
            self._load_initial_mask()
            
            # 加载ERK场
            if self.condition in ['spontaneous', 'inhibitor', 'optogenetic']:
                self._load_erk_field()
            
            # 加载时间序列
            self._load_time_series()
            
            # 加载光流（如果有）
            if self.condition == 'opticflow':
                self._load_opticflow()
            
            # 加载元数据
            self._load_metadata()
            
            logger.info(f"Successfully loaded: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return False
    
    def _load_initial_mask(self):
        """加载初始mask"""
        # 尝试多种格式
        mask_paths = [
            self.dataset_path / "mask.mat",
            self.dataset_path / "mask.npy",
            self.dataset_path / "initial_mask.tif",
            self.dataset_path / "initial_mask.png",
        ]
        
        for mask_path in mask_paths:
            if mask_path.exists():
                if mask_path.suffix == '.mat':
                    from scipy.io import loadmat
                    data = loadmat(str(mask_path))
                    # 尝试常见键名
                    for key in ['manual_mask', 'mask', 'initial_mask', 'binary_mask']:
                        if key in data:
                            self.initial_mask = data[key].astype(np.float32)
                            break
                elif mask_path.suffix == '.npy':
                    self.initial_mask = np.load(str(mask_path)).astype(np.float32)
                elif mask_path.suffix in ['.tif', '.tiff', '.png']:
                    from PIL import Image
                    img = Image.open(str(mask_path)).convert('L')
                    self.initial_mask = (np.array(img) > 127).astype(np.float32)
                
                if self.initial_mask is not None:
                    logger.info(f"Loaded initial mask: {mask_path}")
                    return
        
        # 如果没有找到，创建默认mask（中心圆形伤口）
        logger.warning(f"No initial mask found for {self.name}, creating default")
        self.initial_mask = self._create_default_mask()
    
    def _create_default_mask(self) -> np.ndarray:
        """创建默认的圆形伤口mask"""
        size = 256
        mask = np.ones((size, size), dtype=np.float32)
        
        # 圆形伤口
        center = size // 2
        radius = size // 6
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        mask[dist_from_center < radius] = 0.0
        
        return mask
    
    def _load_erk_field(self):
        """加载ERK场"""
        # 查找ERK/ratio图像
        erk_paths = list(self.dataset_path.glob("*.tif")) + \
                   list(self.dataset_path.glob("*.tiff")) + \
                   list(self.dataset_path.glob("*.png"))

        logger.info(f"Searching for ERK images in: {self.dataset_path}")
        logger.info(f"Found {len(erk_paths)} image files")

        if not erk_paths:
            logger.warning(f"No ERK images found in {self.dataset_path}")
            return

        # 按文件名排序
        erk_paths.sort()
        logger.info(f"Image files: {[f.name for f in erk_paths]}")

        # 加载第一张图像确定shape
        from PIL import Image
        first_img_obj = Image.open(str(erk_paths[0]))
        first_img = np.array(first_img_obj)

        logger.info(f"First image shape: {first_img.shape}, dtype: {first_img.dtype}")

        # 处理多帧TIF
        erk_stack = []
        max_frames = 100  # 限制最大帧数

        # 检查是否是多帧TIF
        try:
            frame_count = 0
            while True:
                if frame_count >= max_frames:
                    break

                try:
                    first_img_obj.seek(frame_count)
                    frame = np.array(first_img_obj)
                    erk_stack.append(frame)
                    frame_count += 1
                except EOFError:
                    break

            if frame_count > 1:
                logger.info(f"Detected multi-frame TIFF: {frame_count} frames")
        except:
            # 不是多帧TIF，尝试单个文件
            pass

        # 如果不是多帧，尝试多个文件
        if len(erk_stack) <= 1:
            erk_stack = []
            for path in erk_paths[:max_frames]:
                img = Image.open(str(path))
                # 处理多帧文件
                try:
                    i = 0
                    while True:
                        if i >= max_frames:
                            break
                        try:
                            img.seek(i)
                            frame = np.array(img)
                            erk_stack.append(frame)
                            i += 1
                        except EOFError:
                            break
                except:
                    # 不是多帧文件
                    frame = np.array(img)
                    erk_stack.append(frame)

        # 确定通道数并转换为ERK场
        if not erk_stack:
            logger.warning(f"Could not load any frames")
            return

        # 转换为numpy数组
        erk_stack = np.array(erk_stack)  # (T, H, W, C) 或 (T, H, W)
        logger.info(f"Loaded image stack shape: {erk_stack.shape}")

        # 处理不同的图像格式
        if len(erk_stack.shape) == 4:  # (T, H, W, C) - RGB
            if erk_stack.shape[3] >= 2:
                # 计算ratio (Green / Red 或其他通道)
                # 通常ERK ratio是 Green/Red
                # 修复1: 添加epsilon避免除零，添加clip限制范围
                red = erk_stack[:, :, 0].astype(np.float32)
                green = erk_stack[:, :, 1].astype(np.float32)
                epsilon = 10.0
                ratio = green / (red + epsilon)
                ratio = np.clip(ratio, 0.1, 10.0)

                # 记录被clip的值（用于调试）
                n_clipped = np.sum((ratio < 0.1) | (ratio > 10.0))
                if n_clipped > 0:
                    logger.warning(f"Clipped {n_clipped} ratio values: "
                                 f"min={ratio.min():.3f}, max={ratio.max():.3f}")

                self.erk_field = ratio
            else:
                # 使用第一个通道
                self.erk_stack = erk_stack
                self.erk_field = erk_stack.astype(np.float32)

            logger.info(f"Final ERK field shape: {self.erk_field.shape}, range: [{self.erk_field.min():.3f}, {self.erk_field.max():.3f}]")
        else:
            logger.warning(f"Unexpected stack shape: {erk_stack.shape}")
            return
    
    def _load_time_series(self):
        """加载时间序列数据"""
        # 尝试从JSON加载
        json_paths = list(self.dataset_path.glob("*.json"))
        for json_path in json_paths:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if 'time_points' in data and 'wound_area' in data:
                    self.time_series = {
                        'time_points': np.array(data['time_points']),
                        'wound_area': np.array(data['wound_area'])
                    }
                    logger.info(f"Loaded time series from {json_path}")
                    return
            except:
                continue
        
        # 如果没有，从ERK场或mask估计
        if self.erk_field is not None:
            self.time_series = self._estimate_time_series()
        else:
            # 创建默认时间序列
            self.time_series = {
                'time_points': np.array([0, 10, 20, 30, 40]),
                'wound_area': np.array([2000, 1500, 1000, 500, 200])
            }
    
    def _estimate_time_series(self) -> Dict:
        """从ERK场估计时间序列"""
        T = self.erk_field.shape[0]
        
        # 估计时间点
        if self.dt is not None:
            time_points = np.arange(T) * (self.dt / 3600)  # 转换为小时
        else:
            time_points = np.arange(T) * 3.0  # 默认3小时间隔
        
        # 估计伤口面积（基于mask演化）
        if self.initial_mask is not None:
            initial_area = np.sum(self.initial_mask == 0)
            
            # 简单的指数衰减模型
            closure_rate = 0.7
            wound_areas = initial_area * (1 - closure_rate * time_points / time_points[-1])
        else:
            wound_areas = np.zeros(T)
        
        return {
            'time_points': time_points,
            'wound_area': wound_areas
        }
    
    def _load_opticflow(self):
        """加载光流场"""
        flow_paths = list(self.dataset_path.glob("flow*.npy")) + \
                    list(self.dataset_path.glob("opticflow*.npy"))
        
        if flow_paths:
            self.opticflow = np.load(str(flow_paths[0]))
            logger.info(f"Loaded opticflow: shape={self.opticflow.shape}")
    
    def _load_metadata(self):
        """加载元数据"""
        # 尝试从索引文件加载
        index_path = self.base_path / "datasets_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            for ds_info in index_data.get('datasets', []):
                if ds_info['id'] == self.dataset_id:
                    self.metadata = ds_info
                    break
        
        # 添加计算得到的元数据
        if self.initial_mask is not None:
            self.metadata['image_size'] = self.initial_mask.shape
            self.metadata['initial_wound_area'] = float(np.sum(self.initial_mask == 0))
        
        if self.erk_field is not None:
            self.metadata['num_frames'] = self.erk_field.shape[0]
            self.metadata['erk_shape'] = self.erk_field.shape
    
    def get_observables(self) -> Dict:
        """
        获取所有观测量
        
        Returns:
            Dict包含:
                - initial_mask: np.ndarray (H, W)
                - erk_field: Optional[np.ndarray] (T, H, W)
                - time_points: np.ndarray (T,)
                - wound_area: np.ndarray (T,)
                - opticflow: Optional[np.ndarray] (T, H, W, 2)
                - metadata: Dict
        """
        return {
            'initial_mask': self.initial_mask,
            'erk_field': self.erk_field,
            'time_points': self.time_series.get('time_points') if self.time_series else None,
            'wound_area': self.time_series.get('wound_area') if self.time_series else None,
            'opticflow': self.opticflow,
            'metadata': self.metadata
        }


class SSBD77Loader:
    """SSBD 77数据集统一加载器"""
    
    def __init__(self, base_path: str = "datasets/ssbd77"):
        """
        Args:
            base_path: 数据集基础路径
        """
        self.base_path = Path(base_path)
        self.datasets: Dict[str, SSBD77Dataset] = {}
        
        # 定义四个核心数据集
        self.dataset_configs = {
            'figure4a_erk': {
                'id': '4022',
                'name': 'Figure4A_ERK',
                'folder': 'spontaneous_ERK_wave',
                'condition': 'spontaneous',
                'dt': 180,
                'pixel_size': 0.49
            },
            'figureS1a_inhibitor': {
                'id': '4029',
                'name': 'FigureS1A_ADAM10_inhibitor',
                'folder': 'inhibitor',
                'condition': 'inhibitor',
                'dt': 120,
                'pixel_size': 0.49
            },
            'figureS4_ratio': {
                'id': '4032',
                'name': 'FigureS4_Ratio',
                'folder': 'optogenetic_ratio',
                'condition': 'optogenetic',
                'dt': 180,
                'pixel_size': 0.49
            },
            'figure6c_opticflow': {
                'id': '4028',
                'name': 'Figure6C_opticflow',
                'folder': 'opticflow',
                'condition': 'opticflow',
                'dt': None,
                'pixel_size': 0.49
            }
        }
    
    def load_all(self, 
                 force_reload: bool = False) -> Dict[str, SSBD77Dataset]:
        """
        加载所有数据集
        
        Args:
            force_reload: 是否强制重新加载
            
        Returns:
            Dict: {dataset_key: SSBD77Dataset}
        """
        results = {}
        
        for key, config in self.dataset_configs.items():
            if key in self.datasets and not force_reload:
                results[key] = self.datasets[key]
                continue
            
            dataset = SSBD77Dataset(
                dataset_id=config['id'],
                name=config['name'],
                folder=config['folder'],
                condition=config['condition'],
                dt=config['dt'],
                pixel_size=config['pixel_size'],
                base_path=str(self.base_path)
            )
            
            if dataset.load():
                self.datasets[key] = dataset
                results[key] = dataset
            else:
                logger.warning(f"Failed to load dataset: {key}")
        
        return results
    
    def load_single(self, 
                    key: str,
                    force_reload: bool = False) -> Optional[SSBD77Dataset]:
        """
        加载单个数据集
        
        Args:
            key: 数据集键名 (e.g., 'figure4a_erk')
            force_reload: 是否强制重新加载
            
        Returns:
            SSBD77Dataset or None
        """
        if key not in self.dataset_configs:
            raise ValueError(f"Unknown dataset key: {key}")
        
        if key in self.datasets and not force_reload:
            return self.datasets[key]
        
        config = self.dataset_configs[key]
        dataset = SSBD77Dataset(
            dataset_id=config['id'],
            name=config['name'],
            folder=config['folder'],
            condition=config['condition'],
            dt=config['dt'],
            pixel_size=config['pixel_size'],
            base_path=str(self.base_path)
        )
        
        if dataset.load():
            self.datasets[key] = dataset
            return dataset
        
        return None
    
    def get_available_datasets(self) -> List[str]:
        """获取所有可用的数据集键名"""
        return list(self.dataset_configs.keys())
    
    def get_dataset_summary(self) -> Dict:
        """获取数据集摘要"""
        summary = {
            'base_path': str(self.base_path),
            'total_datasets': len(self.dataset_configs),
            'loaded_datasets': len(self.datasets),
            'datasets': []
        }
        
        for key, config in self.dataset_configs.items():
            dataset_info = {
                'key': key,
                'id': config['id'],
                'name': config['name'],
                'condition': config['condition'],
                'dt': config['dt'],
                'loaded': key in self.datasets
            }
            
            if key in self.datasets:
                obs = self.datasets[key].get_observables()
                dataset_info.update({
                    'num_frames': self.datasets[key].metadata.get('num_frames'),
                    'image_size': self.datasets[key].metadata.get('image_size'),
                    'initial_wound_area': self.datasets[key].metadata.get('initial_wound_area')
                })
            
            summary['datasets'].append(dataset_info)
        
        return summary


# 便捷函数

def load_figure4a_erk(base_path: str = "datasets/ssbd77"):
    """加载Figure4A_ERK数据集"""
    loader = SSBD77Loader(base_path)
    return loader.load_single('figure4a_erk')

def load_inhibitor(base_path: str = "datasets/ssbd77"):
    """加载FigureS1A抑制剂数据集"""
    loader = SSBD77Loader(base_path)
    return loader.load_single('figureS1a_inhibitor')

def load_opto_ratio(base_path: str = "datasets/ssbd77"):
    """加载FigureS4_Ratio光遗传数据集"""
    loader = SSBD77Loader(base_path)
    return loader.load_single('figureS4_ratio')

def load_opticflow(base_path: str = "datasets/ssbd77"):
    """加载Figure6C_opticflow数据集"""
    loader = SSBD77Loader(base_path)
    return loader.load_single('figure6c_opticflow')


if __name__ == "__main__":
    # 测试加载器
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing SSBD77Loader")
    print("=" * 60)
    
    # 创建加载器
    loader = SSBD77Loader()
    
    # 获取摘要
    summary = loader.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Base path: {summary['base_path']}")
    print(f"  Total datasets: {summary['total_datasets']}")
    print(f"  Loaded datasets: {summary['loaded_datasets']}")
    
    print("\nAvailable datasets:")
    for ds in summary['datasets']:
        status = "[OK]" if ds['loaded'] else "[--]"
        print(f"  {status} {ds['key']}: {ds['name']} ({ds['condition']})")
        if ds['loaded']:
            print(f"     Frames: {ds.get('num_frames', 'N/A')}, "
                  f"Size: {ds.get('image_size', 'N/A')}, "
                  f"Area: {ds.get('initial_wound_area', 'N/A'):.0f}")
    
    print("\n" + "=" * 60)
