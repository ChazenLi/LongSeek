"""
运行参数优化 - 使用网格搜索（更快且更可靠）
"""
import numpy as np
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask
from preprocess.extract_observations import downsample_binary
from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import fit_simple_grid_search, run_simulation, calculate_loss
import matplotlib.pyplot as plt

def run_grid_search_optimization(
    location="L12",
    n_samples=30,  # 增加样本数以获得更好的结果
    downsample_k=4
):
    """
    运行网格搜索优化以找到最佳CA参数

    Args:
        location: 位置标识符
        n_samples: 随机采样次数
        downsample_k: 降采样因子
    """
    print(f"\n{'='*70}")
    print(f"参数优化 - 位置 {location}")
    print(f"{'='*70}\n")

    # 加载时间序列数据
    print("Step 1: 加载时间序列数据...")
    ts_file = f"time_series_{location}.json"
    
    with open(ts_file, 'r') as f:
        ts_data = json.load(f)

    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    
    print(f"  位置: {ts_data['location']}")
    print(f"  时间点: {observed_time}")
    print(f"  伤口面积: {observed_wound_area}")

    # 准备观测数据
    observed = {
        "A_t": observed_wound_area,
    }

    # 加载初始mask
    print("\nStep 2: 加载初始mask...")
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

    # 运行网格搜索
    print(f"\nStep 3: 运行网格搜索优化...")
    print(f"  采样次数: {n_samples}")
    print(f"  降采样因子: {downsample_k}")
    print(f"  预计时间: 约{n_samples * 2}秒...")

    start_time = time.time()
    result = fit_simple_grid_search(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        downsample_k=downsample_k,
        n_samples=n_samples,
    )
    elapsed = time.time() - start_time

    print(f"\n  优化完成，耗时: {elapsed:.1f}秒")
    print(f"  最优损失: {result['best_loss']:.6f}")
    print(f"  最优参数:")
    print(f"    p_move: {result['best_params'].p_move:.4f}")
    print(f"    p_div: {result['best_params'].p_div:.4f}")
    print(f"    alpha: {result['best_params'].alpha:.4f}")
    print(f"    beta: {result['best_params'].beta:.4f}")
    print(f"    edge_bonus: {result['best_params'].edge_bonus:.4f}")
    print(f"    k_time: {result['best_params'].k_time:.4f}")
    print(f"    gamma: {result['best_params'].gamma:.4f}")

    # 使用最优参数运行模拟
    print(f"\nStep 4: 使用最优参数运行模拟...")
    history = run_simulation(
        params=result['best_params'],
        initial_mask=initial_mask,
        num_steps=50,
        downsample_k=downsample_k,
        seed=42,
        observed_time=observed_time,
    )

    simulated_wound_area = np.array([s["wound_area"] for s in history])

    # 归一化
    sim_norm = simulated_wound_area / simulated_wound_area[0]
    obs_norm = observed_wound_area / observed_wound_area[0]

    # 计算指标
    from scipy.interpolate import interp1d
    sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
    f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
    sim_at_obs = f_sim(observed_time)
    
    mse = np.mean((sim_at_obs - obs_norm)**2)
    mae = np.mean(np.abs(sim_at_obs - obs_norm))

    print(f"\n  模拟质量指标:")
    print(f"    MSE: {mse:.6f}")
    print(f"    MAE: {mae:.6f}")

    # 可视化
    print(f"\nStep 5: 生成可视化...")
    output_dir = Path("CA_project/results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 伤口面积对比
    ax1 = axes[0, 0]
    ax1.plot(sim_time_real, sim_norm, 'b-', label='Simulated', linewidth=2, marker='o', markersize=3)
    ax1.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=2, marker='s', markersize=8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized wound area')
    ax1.set_title(f'Wound Area Comparison (MSE={mse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 参数值
    ax2 = axes[0, 1]
    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma']
    param_values = [
        result['best_params'].p_move,
        result['best_params'].p_div,
        result['best_params'].alpha,
        result['best_params'].beta,
        result['best_params'].edge_bonus,
        result['best_params'].k_time,
        result['best_params'].gamma,
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_names)))
    ax2.barh(param_names, param_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Parameter value')
    ax2.set_title('Optimized Parameters')
    ax2.grid(True, alpha=0.3, axis='x')

    # 子图3: 残差分析
    ax3 = axes[1, 0]
    residuals = (sim_at_obs - obs_norm) * 100  # 百分比残差
    ax3.bar(range(len(observed_time)), residuals, color='orange', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlabel('Time point index')
    ax3.set_ylabel('Residual (%)')
    ax3.set_title('Residuals at Observed Time Points')
    ax3.grid(True, alpha=0.3)

    # 子图4: 时间序列对比
    ax4 = axes[1, 1]
    ax4.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=3, marker='o', markersize=10)
    ax4.plot(observed_time, sim_at_obs, 'b-', label='Simulated (interpolated)', linewidth=2, marker='s', markersize=8)
    for i, (x, y_obs, y_sim) in enumerate(zip(observed_time, obs_norm, sim_at_obs)):
        ax4.plot([x, x], [y_obs, y_sim], 'k-', alpha=0.3)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Normalized wound area')
    ax4.set_title('Point-by-Point Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / f"grid_search_opt_{location}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {plot_path}")
    plt.close()

    # 保存结果
    results = {
        "location": location,
        "optimization_method": "Grid Search",
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
        "metrics": {
            "mse": float(mse),
            "mae": float(mae),
        },
        "observed_time": observed_time.tolist(),
        "observed_wound_area": observed_wound_area.tolist(),
        "simulated_wound_area": simulated_wound_area.tolist(),
        "plot_path": str(plot_path),
    }

    results_path = output_dir / f"grid_search_opt_{location}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  保存结果: {results_path}")

    print(f"\n{'='*70}")
    print("参数优化完成！")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    # 运行网格搜索优化（30次采样，比原来的5次更充分）
    results = run_grid_search_optimization(
        location="L12",
        n_samples=30,   # 30次随机采样（比原来的5次多6倍）
        downsample_k=4,
    )

    if results:
        print("\n=== 最终结果 ===")
        print(f"位置: {results['location']}")
        print(f"优化方法: {results['optimization_method']}")
        print(f"采样次数: {results['n_samples']}")
        print(f"优化时间: {results['optimization_time']:.1f}秒")
        print(f"\n最优参数:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v:.4f}")
        print(f"\n质量指标:")
        print(f"  MSE: {results['metrics']['mse']:.6f}")
        print(f"  MAE: {results['metrics']['mae']:.6f}")
