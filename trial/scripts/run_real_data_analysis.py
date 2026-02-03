"""
Real Data Analysis Script - 使用真实数据集运行CA模型
支持4个数据集: SN15, MDCK, Melanoma, TScratch
"""
import numpy as np
import json
import sys
from pathlib import Path
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask, load_measures
from preprocess.extract_observations import extract_frame_statistics, downsample_binary
from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import run_simulation, fit_simple_grid_search
import matplotlib.pyplot as plt

def check_data_availability(dataset_name="SN15"):
    """检查数据集是否可用"""
    # 获取项目根目录（CA_project的父目录）
    project_root = Path(__file__).parent.parent
    base_path = project_root / "CA" / "DATA" / dataset_name / dataset_name

    required_dirs = ["manual_mat"]
    optional_dirs = ["images", "reannotation_mat", "measures.mat"]

    status = {
        "dataset": dataset_name,
        "base_path": str(base_path),
        "exists": base_path.exists(),
        "manual_masks": [],
        "has_measures": False,
    }

    if not status["exists"]:
        return status

    # 检查manual_mat目录
    manual_dir = base_path / "manual_mat"
    if manual_dir.exists():
        mat_files = list(manual_dir.glob("*.mat"))
        status["manual_masks"] = [f.name for f in mat_files if not f.name.startswith("desktop")]

    # 检查measures.mat
    measures_file = base_path / "measures.mat"
    status["has_measures"] = measures_file.exists()

    return status

def analyze_single_dataset(dataset_name="SN15", max_frames=5, num_steps=20):
    """
    分析单个数据集

    Args:
        dataset_name: 数据集名称 (SN15, MDCK, Melanoma, TScratch)
        max_frames: 最多处理多少帧
        num_steps: CA模拟步数
    """
    print(f"\n{'='*70}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*70}\n")

    # Step 1: 检查数据可用性
    print("Step 1: 检查数据可用性...")
    status = check_data_availability(dataset_name)

    if not status["exists"]:
        print(f"  ✗ 数据集路径不存在: {status['base_path']}")
        return None

    print(f"  ✓ 数据集路径存在")
    print(f"  ✓ 找到 {len(status['manual_masks'])} 个mask文件")

    if len(status["manual_masks"]) == 0:
        print(f"  ✗ 没有找到mask文件")
        return None

    if status["has_measures"]:
        print(f"  ✓ 找到measures.mat (质量指标)")

    # Step 2: 加载mask文件
    print(f"\nStep 2: 加载mask文件（前{max_frames}个）...")
    masks_loaded = []
    for i, mask_name in enumerate(status["manual_masks"][:max_frames]):
        mask_path = Path(status["base_path"]) / "manual_mat" / mask_name
        try:
            mask = load_binary_mask(str(mask_path))
            masks_loaded.append({
                "name": mask_name,
                "mask": mask,
                "path": str(mask_path)
            })
            print(f"  [{i+1}] {mask_name}: shape={mask.shape}, cell_coverage={np.mean(mask):.3f}")
        except Exception as e:
            print(f"  [{i+1}] {mask_name}: ✗ 加载失败 - {e}")

    if len(masks_loaded) == 0:
        print("  ✗ 没有成功加载任何mask")
        return None

    # Step 3: 提取观测量
    print(f"\nStep 3: 提取观测量...")
    observations = []
    for item in masks_loaded:
        try:
            stats = extract_frame_statistics(item["mask"])
            stats["frame_name"] = item["name"]
            observations.append(stats)
            print(f"  ✓ {item['name']}: "
                  f"wound_area={stats['wound_area']:.0f}, "
                  f"coverage={stats['cell_coverage']:.3f}, "
                  f"roughness={stats['roughness_perimeter']:.3f}")
        except Exception as e:
            print(f"  ✗ {item['name']}: 提取失败 - {e}")

    if len(observations) < 2:
        print("  ✗ 观测量少于2个，无法进行时间序列分析")
        return None

    # Step 4: 创建时间序列
    print(f"\nStep 4: 创建时间序列...")
    wound_areas = [obs["wound_area"] for obs in observations]
    coverages = [obs["cell_coverage"] for obs in observations]
    frame_names = [obs["frame_name"] for obs in observations]

    print(f"  时间点: {len(observations)} 个")
    print(f"  伤口面积范围: {min(wound_areas):.0f} → {max(wound_areas):.0f}")
    print(f"  覆盖率范围: {min(coverages):.3f} → {max(coverages):.3f}")

    # Step 5: 运行CA模拟
    print(f"\nStep 5: 运行CA模拟...")
    initial_mask = masks_loaded[0]["mask"]

    # 降采样
    downsample_k = 4
    mask_ds = downsample_binary(initial_mask, k=downsample_k, thr=0.5)
    print(f"  降采样: {initial_mask.shape} → {mask_ds.shape} (k={downsample_k})")

    # 使用默认参数运行
    params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, edge_bonus=2.0)
    print(f"  CA参数: p_move={params.p_move}, p_div={params.p_div}, "
          f"alpha={params.alpha}, beta={params.beta}, edge_bonus={params.edge_bonus}")

    start_time = time.time()
    ca = CellOnlyCA(height=mask_ds.shape[0], width=mask_ds.shape[1], params=params)
    ca.initialize_from_mask(initial_mask, k=downsample_k)

    history = ca.run(num_steps=num_steps)
    elapsed = time.time() - start_time

    print(f"  ✓ 模拟完成: {num_steps}步, 耗时{elapsed:.2f}秒 ({elapsed/num_steps*1000:.1f}ms/步)")

    # Step 6: 对比结果
    print(f"\nStep 6: 对比模拟与观测...")
    sim_wound_area = [step["wound_area"] for step in history]
    sim_migrations = [step["migrations"] for step in history]
    sim_divisions = [step["divisions"] for step in history]

    # 归一化
    sim_wound_norm = np.array(sim_wound_area) / sim_wound_area[0]
    obs_wound_norm = np.array(wound_areas) / wound_areas[0]

    print(f"\n  伤口面积变化（归一化）:")
    print(f"    初始: 模拟={sim_wound_norm[0]:.3f}, 观测={obs_wound_norm[0]:.3f}")
    if len(obs_wound_norm) > 1:
        print(f"    中期: 模拟={sim_wound_norm[len(sim_wound_norm)//2]:.3f}, "
              f"观测={obs_wound_norm[len(obs_wound_norm)//2] if len(obs_wound_norm) > 2 else obs_wound_norm[-1]:.3f}")
    print(f"    最终: 模拟={sim_wound_norm[-1]:.3f}, 观测={obs_wound_norm[-1]:.3f}")

    # 计算MSE
    mse = None
    obs_time_indices = None
    if len(obs_wound_norm) >= 2:
        # 插值到观测时间点
        from scipy.interpolate import interp1d
        sim_time = np.linspace(0, len(sim_wound_norm)-1, len(sim_wound_norm))
        obs_time_indices = np.linspace(0, len(sim_wound_norm)-1, len(obs_wound_norm))
        f = interp1d(sim_time, sim_wound_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
        sim_interp = f(obs_time_indices)

        mse = np.mean((sim_interp - obs_wound_norm)**2)
        print(f"    MSE: {mse:.4f}")

    # Step 7: 可视化
    print(f"\nStep 7: 生成可视化...")
    output_dir = Path("CA_project/results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 子图1: 伤口面积对比
    ax1 = axes[0, 0]
    ax1.plot(sim_wound_norm, 'b-', label='Simulated', linewidth=2, marker='o', markersize=4)
    ax1.plot(obs_time_indices, obs_wound_norm, 'r--', label='Observed', linewidth=2, marker='s', markersize=6)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Normalized wound area')
    ax1.set_title('Wound Area: Simulated vs Observed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 细胞活动
    ax2 = axes[0, 1]
    ax2.plot(sim_migrations, 'b-', label='Migrations', linewidth=2, alpha=0.7)
    ax2.plot(sim_divisions, 'r-', label='Divisions', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Count')
    ax2.set_title('Cellular Activities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 初始mask
    ax3 = axes[1, 0]
    im3 = ax3.imshow(masks_loaded[0]["mask"], cmap='RdYlBu_r')
    ax3.set_title(f'Initial Mask: {masks_loaded[0]["name"]}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 子图4: 最终CA状态
    ax4 = axes[1, 1]
    final_grid = ca.grid
    im4 = ax4.imshow(final_grid, cmap='RdYlBu_r')
    ax4.set_title(f'Final CA State (Step {num_steps})')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    plt.tight_layout()

    plot_path = output_dir / f"real_data_{dataset_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存图像: {plot_path}")
    plt.close()

    # Step 8: 保存结果
    results = {
        "dataset": dataset_name,
        "num_frames": len(masks_loaded),
        "num_steps": num_steps,
        "downsample_k": downsample_k,
        "ca_params": {
            "p_move": params.p_move,
            "p_div": params.p_div,
            "alpha": params.alpha,
            "beta": params.beta,
            "edge_bonus": params.edge_bonus,
        },
        "frames": [
            {
                "name": obs["frame_name"],
                "wound_area": float(obs["wound_area"]),
                "cell_coverage": float(obs["cell_coverage"]),
                "roughness": float(obs["roughness_perimeter"]),
            }
            for obs in observations
        ],
        "simulation": {
            "wound_area": [float(w) for w in sim_wound_area],
            "migrations": [int(m) for m in sim_migrations],
            "divisions": [int(d) for d in sim_divisions],
            "mse": float(mse) if mse is not None else None,
        },
        "plot_path": str(plot_path),
    }

    results_path = output_dir / f"real_data_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ 保存结果: {results_path}")

    print(f"\n{'='*70}")
    print(f"数据集 {dataset_name} 分析完成！")
    print(f"{'='*70}\n")

    return results

def main():
    """主函数：分析所有可用数据集"""
    print("="*70)
    print("真实数据集CA模型分析")
    print("="*70)

    datasets = ["SN15", "MDCK", "Melanoma", "TScratch"]
    results = {}

    for dataset in datasets:
        try:
            result = analyze_single_dataset(
                dataset_name=dataset,
                max_frames=5,  # 每个数据集最多5帧
                num_steps=30   # 模拟30步
            )
            if result:
                results[dataset] = result
        except Exception as e:
            print(f"\n✗ 分析 {dataset} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"成功分析的数据集: {len(results)}/4")
    for dataset, result in results.items():
        print(f"\n{dataset}:")
        print(f"  - 处理帧数: {result['num_frames']}")
        print(f"  - 模拟步数: {result['num_steps']}")
        print(f"  - MSE: {result['simulation']['mse']:.4f}" if result['simulation']['mse'] else "  - MSE: N/A")
        print(f"  - 结果文件: {result['plot_path']}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
