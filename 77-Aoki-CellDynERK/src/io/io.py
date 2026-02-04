"""
IO 模块 - 读写图像栈数据

支持 TIFF 读取和 Zarr 写入（参考 plan.md 第 2 节）
"""

import json
import numpy as np
import tifffile
import zarr
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class SequenceMetadata:
    """序列元数据（参考 plan.md: 可追溯性底座）"""
    dt_seconds: float  # 时间步长（秒）
    um_per_px: float  # 像素尺度（微米/像素）
    source_url: str  # 数据来源 URL
    dataset_name: str  # 数据集名称
    condition: str  # 实验条件
    shape: Optional[Tuple[int, ...]] = None  # (T, H, W) 或 (T, C, H, W)
    dtype: Optional[str] = None
    commit_hash: Optional[str] = None  # git commit（可选）

    def to_dict(self) -> dict:
        return {
            "dt_seconds": self.dt_seconds,
            "um_per_px": self.um_per_px,
            "source_url": self.source_url,
            "dataset_name": self.dataset_name,
            "condition": self.condition,
            "shape": self.shape,
            "dtype": self.dtype,
            "commit_hash": self.commit_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceMetadata":
        return cls(
            dt_seconds=d["dt_seconds"],
            um_per_px=d["um_per_px"],
            source_url=d["source_url"],
            dataset_name=d["dataset_name"],
            condition=d["condition"],
            shape=d.get("shape"),
            dtype=d.get("dtype"),
            commit_hash=d.get("commit_hash"),
        )


def read_tiff_stack(path: Path) -> np.ndarray:
    """
    读取 TIFF 栈

    Args:
        path: TIFF 文件路径

    Returns:
        (T, H, W) 或 (T, C, H, W) 数组
    """
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()
    return stack


def write_zarr(
    data: np.ndarray,
    path: Path,
    chunks: Optional[Tuple[int, ...]] = None,
    compressor: Any = None,
) -> None:
    """
    写入 Zarr 文件

    Args:
        data: 数据数组
        path: 输出路径（不含扩展名）
        chunks: 分块大小，默认 (1, H, W) 或 (1, C, H, W)
        compressor: 压缩器，默认 Blosc
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        # 默认时间维度分块
        chunks = (1,) + data.shape[1:]

    if compressor is None:
        from numcodecs import Blosc
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

    z = zarr.open_array(
        path,
        mode='w',
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        compressor=compressor,
    )
    z[:] = data


def read_zarr(path: Path) -> np.ndarray:
    """
    读取 Zarr 文件

    Args:
        path: Zarr 目录路径

    Returns:
        数据数组
    """
    return zarr.open_array(path, mode='r')[:]


def write_meta(meta: SequenceMetadata, path: Path) -> None:
    """
    写入元数据 JSON

    Args:
        meta: 元数据对象
        path: 输出路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)


def read_meta(path: Path) -> SequenceMetadata:
    """
    读取元数据 JSON

    Args:
        path: JSON 文件路径

    Returns:
        元数据对象
    """
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return SequenceMetadata.from_dict(d)


def write_observables(observables: dict, path: Path) -> None:
    """
    写入观测量 JSON

    Args:
        observables: 观测量字典
        path: 输出路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换 numpy 类型为 Python 原生类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert(observables), f, indent=2, ensure_ascii=False)


def read_observables(path: Path) -> dict:
    """
    读取观测量 JSON

    Args:
        path: JSON 文件路径

    Returns:
        观测量字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ProcessedSequence:
    """
    预处理后的序列封装类

    目录结构（参考 plan.md）:
    <seq_name>/
        C_obs.zarr/         # (T,H,W) float32 - ERK 活性场
        occ_mask.zarr/      # (T,H,W) uint8  - 1=cell 0=wound/empty
        meta.json           # 元数据
        observables.json    # 观测量
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._meta: Optional[SequenceMetadata] = None
        self._observables: Optional[dict] = None

    @property
    def meta(self) -> SequenceMetadata:
        if self._meta is None:
            self._meta = read_meta(self.root / "meta.json")
        return self._meta

    @property
    def observables(self) -> dict:
        if self._observables is None:
            obs_path = self.root / "observables.json"
            if obs_path.exists():
                self._observables = read_observables(obs_path)
            else:
                self._observables = {}
        return self._observables

    def load_C_obs(self) -> np.ndarray:
        """加载 ERK 活性场"""
        return read_zarr(self.root / "C_obs.zarr")

    def load_occ_mask(self) -> np.ndarray:
        """加载占据掩膜"""
        return read_zarr(self.root / "occ_mask.zarr")

    def save_C_obs(self, data: np.ndarray) -> None:
        """保存 ERK 活性场"""
        write_zarr(data, self.root / "C_obs.zarr")

    def save_occ_mask(self, data: np.ndarray) -> None:
        """保存占据掩膜"""
        write_zarr(data, self.root / "occ_mask.zarr", chunks=(1, data.shape[1], data.shape[2]))

    def save_meta(self, meta: SequenceMetadata) -> None:
        """保存元数据"""
        write_meta(meta, self.root / "meta.json")
        self._meta = meta

    def save_observables(self, observables: dict) -> None:
        """保存观测量"""
        write_observables(observables, self.root / "observables.json")
        self._observables = observables

    def exists(self) -> bool:
        """检查序列是否已存在"""
        return (self.root / "meta.json").exists()
