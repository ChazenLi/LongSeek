"""
可视化真实mask以理解数据
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask

def visualize_mask(dataset_name="SN15", mask_index=0):
    """可视化一个mask"""

    # 路径
    project_root = Path(__file__).parent.parent
    base_path = project_root / "CA" / "DATA" / dataset_name / dataset_name
    manual_dir = base_path / "manual_mat"

    # 获取mask文件
    mat_files = sorted([f for f in manual_dir.glob("*.mat") if not f.name.startswith("desktop")])

    if mask_index >= len(mat_files):
        print(f"错误: mask_index {mask_index} 超出范围 (0-{len(mat_files)-1})")
        return

    mat_file = mat_files[mask_index]
    print(f"加载: {mat_file.name}")

    # 加载mask
    mask = load_binary_mask(str(mat_file))

    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  唯一值: {np.unique(mask)}")
    print(f"  值=0的像素数 (wound?): {np.sum(mask == 0)}")
    print(f"  值=1的像素数 (cell?): {np.sum(mask == 1)}")
    print(f"  总像素数: {mask.size}")
    print(f"  覆盖率: {np.mean(mask):.3f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始mask
    im1 = axes[0].imshow(mask, cmap='RdYlBu_r')
    axes[0].set_title(f'Original Mask\n{mat_file.name}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 伤口区域 (假设mask=0是伤口)
    wound = (mask == 0)
    im2 = axes[1].imshow(wound, cmap='Reds')
    axes[1].set_title(f'Wound Area (mask==0)\n{np.sum(wound)} pixels')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # 细胞区域 (假设mask=1是细胞)
    cells = (mask == 1)
    im3 = axes[2].imshow(cells, cmap='Blues')
    axes[2].set_title(f'Cell Area (mask==1)\n{np.sum(cells)} pixels')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    # 保存
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"mask_visualization_{dataset_name}_{mask_index}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图像已保存: {output_path}")

    # 不显示窗口，只保存
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SN15", help="数据集名称")
    parser.add_argument("--index", type=int, default=0, help="mask索引")
    args = parser.parse_args()

    visualize_mask(args.dataset, args.index)
