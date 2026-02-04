"""
ERK Dataset Loader - 统一ERK数据集加载器 (重构版)

支持：
- Figure4A: spontaneous (dt=180s, 60帧)
- FigureS1A: inhibitor (dt=120s)
- FigureS4: optogenetic (dt=180s)
- Figure6C: opticflow (速度场)

核心特性：
1. 统一接口加载所有ERK相关数据集
2. 支持60帧完整时间序列数据
3. 自动解析metadata (dt, pixel_size, condition)
4. 预计算三层观测量
5. 时滞缓冲区支持

作者: IF-CA Team
日期: 2026-02-04
版本: 2.0 (Refactored)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ERKDatasetMetadata:
    """ERK数据集元数据"""
    dataset_key: str              # 数据集键名
    condition: str                # 条件类型
    dt_seconds: float             # 时间间隔(秒)
    pixel_size_um: float          # 像素大小(微米)
    num_frames: int               # 帧数
    shape: Tuple[int, int, int]   # (T, H, W)
    has_opticflow: bool = False   # 是否有光流数据


@dataclass
class ERKDataset:
    """
    ERK数据集容器

    包含：
    - erk_field: (T, H, W) ERK场序列 - 支持60帧完整数据
    - mask: (H, W) 细胞mask
    - optic_flow: Optional[(T, H, W, 2)] 速度场
    - metadata: ERKDatasetMetadata 元数据
    - time_points_hours: (T,) 时间点(小时)
    - observables: 三层观测量字典
    """

    erk_field: np.ndarray                        # (T, H, W) ERK场序列
    mask: np.ndarray                             # (H, W) 细胞mask
    metadata: ERKDatasetMetadata                 # 元数据
    time_points_hours: np.ndarray                # (T,) 时间点
    optic_flow: Optional[np.ndarray] = None      # (T, H, W, 2) 光流
    observables: Dict[str, np.ndarray] = field(default_factory=dict)  # 三层观测


class ERKDatasetLoader:
    """
    统一ERK数据集加载器 (重构版)

    用法:
        loader = ERKDatasetLoader(base_path='datasets')
        data = loader.load('figure4a_erk')
        # 或使用别名
        data = loader.load('spontaneous')

    数据集配置:
    - figure4a_erk / spontaneous: 自发ERK波 (dt=180s)
    - figureS1a_inhibitor / inhibitor: 抑制剂 (dt=120s)
    - figureS4_ratio / opto: 光遗传 (dt=180s)
    - figure6c_opticflow / opticflow: 光流
    """

    # 数据集配置
    DATASET_CONFIGS = {
        'figure4a_erk': {
            'folder': 'ssbd77/spontaneous_ERK_wave',
            'condition': 'spontaneous',
            'dt_seconds': 180.0,
            'pixel_size_um': 0.49,
            'expected_frames': 60
        },
        'figureS1a_inhibitor': {
            'folder': 'ssbd77/inhibitor',
            'condition': 'inhibitor',
            'dt_seconds': 120.0,
            'pixel_size_um': 0.49,
            'expected_frames': 60
        },
        'figureS4_ratio': {
            'folder': 'ssbd77/optogenetic_ratio',
            'condition': 'optogenetic',
            'dt_seconds': 180.0,
            'pixel_size_um': 0.49,
            'expected_frames': 41
        },
        'figure6c_opticflow': {
            'folder': 'ssbd77/opticflow',
            'condition': 'opticflow',
            'dt_seconds': 180.0,
            'pixel_size_um': 0.49,
            'expected_frames': 31
        }
    }

    # 别名
    ALIASES = {
        'spontaneous': 'figure4a_erk',
        'inhibitor': 'figureS1a_inhibitor',
        'opto': 'figureS4_ratio',
        'optogenetic': 'figureS4_ratio',
        'opticflow': 'figure6c_opticflow',
    }

    def __init__(self, base_path: str = "datasets"):
        """
        Args:
            base_path: 数据集基础路径
        """
        self.base_path = Path(base_path)
        self._cache: Dict[str, ERKDataset] = {}

    def resolve_key(self, dataset_key: str) -> str:
        """解析数据集键名（处理别名）"""
        return self.ALIASES.get(dataset_key, dataset_key)

    def list_datasets(self) -> List[str]:
        """列出所有可用数据集"""
        return list(self.DATASET_CONFIGS.keys())

    def load(self, dataset_key: str,
             force_reload: bool = False,
             compute_observables: bool = True) -> ERKDataset:
        """
        加载数据集

        Args:
            dataset_key: 数据集键名（支持别名）
            force_reload: 是否强制重新加载
            compute_observables: 是否计算观测量

        Returns:
            ERKDataset对象
        """
        # 解析别名
        resolved_key = self.resolve_key(dataset_key)

        # 检查缓存
        if resolved_key in self._cache and not force_reload:
            logger.info(f"Loading from cache: {resolved_key}")
            return self._cache[resolved_key]

        # 验证数据集键名
        if resolved_key not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys()) + list(self.ALIASES.keys())
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {available}")

        config = self.DATASET_CONFIGS[resolved_key]
        dataset_path = self.base_path / config['folder']

        logger.info(f"Loading dataset: {resolved_key} from {dataset_path}")

        # 1. 加载ERK场
        erk_field = self._load_erk_field(dataset_path)

        # 2. 加载mask
        mask = self._load_mask(dataset_path, erk_field.shape[1:])

        # 3. 加载光流（如果有）
        optic_flow = None
        if config['condition'] == 'opticflow':
            optic_flow = self._load_opticflow(dataset_path)

        # 4. 创建元数据
        metadata = ERKDatasetMetadata(
            dataset_key=resolved_key,
            condition=config['condition'],
            dt_seconds=config['dt_seconds'],
            pixel_size_um=config['pixel_size_um'],
            num_frames=erk_field.shape[0],
            shape=erk_field.shape,
            has_opticflow=optic_flow is not None
        )

        # 5. 计算时间点
        time_points_hours = np.arange(metadata.num_frames) * \
                           (metadata.dt_seconds / 3600.0)

        # 6. 创建数据集对象
        dataset = ERKDataset(
            erk_field=erk_field.astype(np.float32),
            mask=mask.astype(np.float32),
            metadata=metadata,
            time_points_hours=time_points_hours,
            optic_flow=optic_flow
        )

        # 7. 计算观测量
        if compute_observables:
            dataset.observables = self._compute_observables(dataset)

        # 8. 缓存
        self._cache[resolved_key] = dataset

        logger.info(f"Successfully loaded {resolved_key}: "
                   f"shape={erk_field.shape}, frames={metadata.num_frames}")

        return dataset

    def load_all(self,
                 force_reload: bool = False,
                 compute_observables: bool = True) -> Dict[str, ERKDataset]:
        """加载所有数据集"""
        results = {}
        for key in self.DATASET_CONFIGS:
            try:
                results[key] = self.load(key, force_reload, compute_observables)
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
        return results

    def get_summary(self) -> Dict:
        """获取数据集摘要"""
        return {
            'base_path': str(self.base_path),
            'num_datasets': len(self.DATASET_CONFIGS),
            'cached': len(self._cache),
            'datasets': list(self.DATASET_CONFIGS.keys()),
            'aliases': self.ALIASES
        }

    def _load_erk_field(self, dataset_path: Path) -> np.ndarray:
        """加载ERK场 - 支持60帧完整数据"""
        # 1. 尝试从NPZ加载
        npz_files = list(dataset_path.rglob("*.npz"))
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                # 查找ERK相关键
                for key in data.keys():
                    if 'erk' in key.lower() or 'ratio' in key.lower() or 'field' in key.lower():
                        field = data[key]
                        if field.ndim == 3:  # (T, H, W)
                            logger.info(f"Loaded ERK from {npz_file.name}: {field.shape}")
                            return field
            except Exception as e:
                continue

        # 2. 尝试从TIF加载
        tif_files = list(dataset_path.rglob("*.tif")) + \
                   list(dataset_path.rglob("*.tiff"))
        if tif_files:
            return self._load_erk_from_tif(tif_files)

        raise FileNotFoundError(f"No ERK field found in {dataset_path}")

    def _load_erk_from_tif(self, tif_files: List[Path]) -> np.ndarray:
        """从TIF文件加载ERK场"""
        from PIL import Image

        tif_files.sort()
        all_frames = []

        for tif_file in tif_files:
            try:
                with Image.open(tif_file) as img:
                    # 读取多帧TIFF
                    try:
                        for i in range(img.n_frames):
                            img.seek(i)
                            frame = np.array(img, dtype=np.float32)
                            all_frames.append(frame)
                    except (AttributeError, EOFError):
                        # 单帧
                        frame = np.array(img, dtype=np.float32)
                        all_frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to load {tif_file}: {e}")
                continue

        if not all_frames:
            raise ValueError("No valid frames in TIF files")

        # 转换为numpy数组
        erk_stack = np.array(all_frames)

        # 处理RGB图像：计算G/R比值
        if erk_stack.ndim == 4 and erk_stack.shape[3] >= 2:
            red = erk_stack[:, :, :, 0].astype(np.float32)
            green = erk_stack[:, :, :, 1].astype(np.float32)
            epsilon = 10.0
            ratio = green / (red + epsilon)
            ratio = np.clip(ratio, 0.1, 10.0)
            erk_field = ratio
        else:
            erk_field = erk_stack.astype(np.float32)

        logger.info(f"Loaded ERK field from TIF: {erk_field.shape} ({erk_field.shape[0]} frames)")
        return erk_field

    def _load_mask(self, dataset_path: Path,
                   expected_shape: Tuple[int, int]) -> np.ndarray:
        """加载mask"""
        # 尝试多种格式
        mask_files = list(dataset_path.rglob("mask.*")) + \
                    list(dataset_path.rglob("*mask*.*"))

        for mask_file in mask_files:
            if mask_file.suffix in ['.npy', '.mat', '.tif', '.tiff', '.png']:
                try:
                    if mask_file.suffix == '.npy':
                        mask = np.load(mask_file)
                    elif mask_file.suffix == '.mat':
                        from scipy.io import loadmat
                        data = loadmat(str(mask_file))
                        mask = None
                        for key in ['mask', 'manual_mask', 'binary_mask', 'initial_mask']:
                            if key in data:
                                mask = data[key]
                                break
                        if mask is None:
                            continue
                    else:
                        from PIL import Image
                        img = Image.open(mask_file).convert('L')
                        mask = (np.array(img) > 127).astype(np.float32)

                    # 调整尺寸
                    if mask.shape != expected_shape:
                        from skimage.transform import resize
                        mask = resize(mask, expected_shape,
                                     preserve_range=True,
                                     anti_aliasing=True).astype(np.float32)

                    logger.info(f"Loaded mask from {mask_file.name}: {mask.shape}")
                    return mask

                except Exception as e:
                    logger.warning(f"Failed to load mask from {mask_file}: {e}")
                    continue

        # 创建默认mask
        logger.warning(f"No mask found, creating default")
        return self._create_default_mask(expected_shape)

    def _create_default_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """创建默认mask"""
        mask = np.ones(shape, dtype=np.float32)
        # 中心圆形"伤口"
        H, W = shape
        center_y, center_x = H // 2, W // 2
        radius = min(H, W) // 8

        y, x = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[dist_from_center < radius] = 0.0

        return mask

    def _load_opticflow(self, dataset_path: Path) -> Optional[np.ndarray]:
        """加载光流场"""
        flow_files = list(dataset_path.rglob("*flow*.npz")) + \
                    list(dataset_path.rglob("*flow*.npy"))

        for flow_file in flow_files:
            try:
                flow = np.load(flow_file)
                if flow.ndim == 4 and flow.shape[-1] == 2:  # (T, H, W, 2)
                    logger.info(f"Loaded optic flow from {flow_file.name}: {flow.shape}")
                    return flow
            except Exception as e:
                continue

        return None

    def _compute_observables(self, dataset: ERKDataset) -> Dict[str, np.ndarray]:
        """
        计算三层观测量

        调用 three_layer_observables 模块
        """
        from .three_layer_observables import compute_three_layer_observables

        observables = compute_three_layer_observables(
            erk_field=dataset.erk_field,
            mask=dataset.mask,
            optic_flow=dataset.optic_flow,
            dt_seconds=dataset.metadata.dt_seconds,
            pixel_size_um=dataset.metadata.pixel_size_um
        )

        return observables


# 便捷函数
def load_erk_dataset(dataset_key: str,
                    base_path: str = "datasets") -> ERKDataset:
    """加载ERK数据集的便捷函数"""
    loader = ERKDatasetLoader(base_path)
    return loader.load(dataset_key)


def load_all_erk_datasets(base_path: str = "datasets") -> Dict[str, ERKDataset]:
    """加载所有ERK数据集的便捷函数"""
    loader = ERKDatasetLoader(base_path)
    return loader.load_all()


if __name__ == "__main__":
    # 测试加载器
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing ERKDatasetLoader v2.0")
    print("=" * 60)

    loader = ERKDatasetLoader()

    # 打印摘要
    summary = loader.get_summary()
    print(f"\nDataset Summary:")
    print(f"  Base path: {summary['base_path']}")
    print(f"  Total datasets: {summary['num_datasets']}")

    print("\nAvailable datasets:")
    for key in summary['datasets']:
        config = loader.DATASET_CONFIGS[key]
        print(f"  - {key}: {config['condition']}, dt={config['dt_seconds']}s")

    print("\nAliases:")
    for alias, key in summary['aliases'].items():
        print(f"  {alias} -> {key}")

    print("\n" + "=" * 60)
