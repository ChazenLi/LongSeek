"""
快速参数优化 - 使用较少采样以快速获得结果
"""
import numpy as np
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask
from inference.fit_bo import fit_simple_grid_search, run_simulation
import matplotlib.pyplot as plt

def quick_optimization(location="L12", n_samples=10, downsample_k=4):
    """快速优化"""
    print(f"\n{'='*70}")
    print(f"快速参数优化 - 位置 {location} ({n_samples}次采样)")
    print(f"{'='*70}\n")

    # 加载数据
    print("加载时间序列数据...")
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)

    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    observed = {"A_t": observed_wound_area}
    
    print(f"  时间点: {observed_time}")
    print(f"  伤口面积: {observed_wound_area}")

    # 加载mask
    print("\n加载初始mask...")
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    first_frame_id = ts_data["frame_ids"][0]
    row = manifest[manifest["frame_id"] == first_frame_id]
    mask_path = row["manual_mask_path"].values[0]
    
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        project_root = Path(__file__).parent.parent
        mask_path = str(project_root / mask_path)
    
    initial_mask = load_binary_mask(mask_path)
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Cell coverage: {np.mean(initial_mask):.3f}")

    # 运行优化
    print(f"\n运行网格搜索优化 ({n_samples}次采样)...")
    print(f"预计时间: 约{n_samples * 1.5}秒...")
    
    start = time.time()
    result = fit_simple_grid_search(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        downsample_k=downsample_k,
        n_samples=n_samples,
    )
    elapsed = time.time() - start

    print(f"\n优化完成，耗时: {elapsed:.1f}秒")
    print(f"最优损失: {result['best_loss']:.6f}")
    print(f"\n最优参数:")
    print(f"  p_move: {result['best_params'].p_move:.4f}")
    print(f"  p_div: {result['best_params'].p_div:.4f}")
    print(f"  alpha: {result['best_params'].alpha:.4f}")
    print(f"  beta: {result['best_params'].beta:.4f}")
    print(f"  edge_bonus: {result['best_params'].edge_bonus:.4f}")
    print(f"  k_time: {result['best_params'].k_time:.4f}")
    print(f"  gamma: {result['best_params'].gamma:.4f}")

    # 使用最优参数运行完整模拟
    print(f"\n使用最优参数运行完整模拟...")
    history = run_simulation(
        params=result['best_params'],
        initial_mask=initial_mask,
        num_steps=50,
        downsample_k=downsample_k,
        seed=42,
        observed_time=observed_time,
    )

    simulated_wound_area = np.array([s["wound_area"] for s in history])
    sim_norm = simulated_wound_area / simulated_wound_area[0]
    obs_norm = observed_wound_area / observed_wound_area[0]

    # 计算MSE
    from scipy.interpolate import interp1d
    sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
    f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
    sim_at_obs = f_sim(observed_time)
    mse = np.mean((sim_at_obs - obs_norm)**2)

    print(f"\n模拟质量:")
    print(f"  MSE: {mse:.6f}")
    print(f"  观测伤口面积: {obs_wound_area}")
    print(f"  模拟伤口面积 (插值到观测点): {sim_at_obs}")

    # 可视化
    print(f"\n生成可视化...")
    output_dir = Path("CA_project/results")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(sim_time_real, sim_norm, 'b-', label='Simulated', linewidth=2, marker='o', markersize=3)
    plt.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=2, marker='s', markersize=8)
    plt.xlabel('Time')
    plt.ylabel('Normalized wound area')
    plt.title(f'Quick Optimization: {location} (MSE={mse:.4f}, {n_samples} samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = output_dir / f"quick_opt_{location}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"保存图像: {plot_path}")
    plt.close()

    # 保存结果
    results = {
        "location": location,
        "n_samples": n_samples,
        "optimization_time": elapsed,
        "best_params": {
            "p_move": float(result['best_params'].p_move),
            "p_div": float(result['best_params'].p_div),
            "alpha": float(result['best_params'].alpha),
            "beta": float(result['best_params'].beta),
            "edge_bonus": float(result['best_params'].edge_bonus),
            "k_time": float(result['best_params'].k_time),
            "gamma": float(result['best_params'].gamma),
        },
        "best_loss": float(result['best_loss']),
        "mse": float(mse),
        "observed_time": observed_time.tolist(),
        "observed_wound_area": observed_wound_area.tolist(),
        "simulated_wound_area": simulated_wound_area.tolist(),
        "plot_path": str(plot_path),
    }

    results_path = output_dir / f"quick_opt_{location}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"保存结果: {results_path}")

    print(f"\n{'='*70}")
    print("优化完成！")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    # 快速优化（10次采样）
    results = quick_optimization(
        location="L12",
        n_samples=10,
        downsample_k=4,
    )

    if results:
        print("\n=== 最终结果 ===")
        print(f"位置: {results['location']}")
        print(f"采样次数: {results['n_samples']}")
        print(f"优化时间: {results['optimization_time']:.1f}秒")
        print(f"MSE: {results['mse']:.6f}")
        print(f"\n最优参数:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v:.4f}")
