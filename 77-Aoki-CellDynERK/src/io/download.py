"""
数据下载模块 - 从 SSBD 下载 77-Aoki-CellDynERK 数据集

参考 plan.md 第 1 节：数据获取
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# SSBD 数据集直链（来自 plan.md）
SSBD_BASE_URL = "https://ssbd.riken.jp/data/77-Aoki-CellDynERK/source"

DATASETS = {
    "Figure4A_ERK": {
        "url": f"{SSBD_BASE_URL}/Figure4A_ERK.zip",
        "dt": 180,  # 秒
        "um_per_px": 0.49,
        "condition": "spontaneous",
        "description": "自发 ERK 波"
    },
    "FigureS1A_ADAM10_inhibitor": {
        "url": f"{SSBD_BASE_URL}/FigureS1A_ADAM10_inhibitor.zip",
        "dt": 120,  # 秒
        "um_per_px": 0.65,
        "condition": "inhibitor",
        "description": "ADAM10 抑制条件"
    },
    "FigureS4_Ratio": {
        "url": f"{SSBD_BASE_URL}/FigureS4_Ratio.zip",
        "dt": 180,  # 秒
        "um_per_px": 1.48,
        "condition": "optogenetic",
        "description": "光遗传诱导 ERK 激活"
    },
    "Figure6C_opticflow": {
        "url": f"{SSBD_BASE_URL}/Figure6C_opticflow.zip",
        "dt": 120,  # 秒
        "um_per_px": 1.48,
        "condition": "opticflow",
        "description": "官方光流 montage"
    }
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        dest_path: 目标路径
        chunk_size: 下载块大小
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    解压 zip 文件

    Args:
        zip_path: zip 文件路径
        extract_to: 解压目标目录
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(desc=f"Extracting {zip_path.name}", total=len(file_list)) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)


def download_dataset(
    dataset_name: str,
    raw_dir: Path,
    force: bool = False
) -> Path:
    """
    下载单个数据集

    Args:
        dataset_name: 数据集名称（DATASETS 的键）
        raw_dir: 原始数据目录
        force: 是否强制重新下载

    Returns:
        解压后的数据目录路径
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[dataset_name]
    url = info["url"]

    # 文件路径
    zip_filename = url.split('/')[-1]
    zip_path = raw_dir / zip_filename
    extract_path = raw_dir / dataset_name

    # 检查是否已存在
    if extract_path.exists() and not force:
        print(f"Dataset {dataset_name} already exists at {extract_path}")
        return extract_path

    # 下载
    print(f"Downloading {dataset_name} from {url}")
    download_file(url, zip_path)

    # 解压
    print(f"Extracting to {extract_path}")
    extract_zip(zip_path, extract_path)

    # 保存元数据
    meta_path = extract_path / "_download_meta.json"
    import json
    with open(meta_path, 'w') as f:
        json.dump({
            "dataset_name": dataset_name,
            "source_url": url,
            "dt_seconds": info["dt"],
            "um_per_px": info["um_per_px"],
            "condition": info["condition"],
            "description": info["description"],
        }, f, indent=2)

    return extract_path


def download_all_datasets(
    raw_dir: Optional[Path] = None,
    datasets: Optional[list[str]] = None,
    force: bool = False
) -> dict[str, Path]:
    """
    下载多个数据集

    Args:
        raw_dir: 原始数据目录，默认为 data/raw/77-Aoki-CellDynERK
        datasets: 要下载的数据集列表，None 表示下载全部
        force: 是否强制重新下载

    Returns:
        数据集名称到路径的映射
    """
    if raw_dir is None:
        raw_dir = Path("data/raw/77-Aoki-CellDynERK")

    if datasets is None:
        datasets = list(DATASETS.keys())

    results = {}
    for name in datasets:
        try:
            path = download_dataset(name, raw_dir, force)
            results[name] = path
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    return results


def main():
    """命令行入口"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Download SSBD 77-Aoki-CellDynERK datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["Figure4A_ERK", "FigureS1A_ADAM10_inhibitor", "FigureS4_Ratio"],
        help="Datasets to download"
    )
    parser.add_argument("--raw-dir", default="data/raw/77-Aoki-CellDynERK", help="Raw data directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")

    args = parser.parse_args()

    if "all" in args.datasets:
        datasets = list(DATASETS.keys())
    else:
        datasets = args.datasets

    raw_dir = Path(args.raw_dir)
    results = download_all_datasets(raw_dir, datasets, args.force)

    print("\nDownload summary:")
    for name, path in results.items():
        print(f"  {name}: {path}")

    # 保存索引
    index_path = raw_dir / "_datasets_index.json"
    with open(index_path, 'w') as f:
        json.dump({
            "datasets": {
                name: {
                    "path": str(path),
                    "info": DATASETS[name]
                }
                for name in results.keys()
            },
            "total_downloaded": len(results)
        }, f, indent=2)


if __name__ == "__main__":
    main()
