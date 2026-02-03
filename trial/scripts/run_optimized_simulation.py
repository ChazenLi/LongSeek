"""
参数优化结果总结和使用优化参数运行模拟
"""
import numpy as np
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask
from inference.fit_bo import run_simulation, calculate_loss
import matplotlib.pyplot as plt

def run_optimized_simulation(location="L12", use_best_params=True):
    """
    使用优化后的参数运行完整模拟
    
    Args:
        location: 位置标识符
        use_best_params: 是否使用已保存的最佳参数
    """
    print(f"\n{'='*70}")
    print(f"优化参数模拟 - 位置 {location}")
    print(f"{'='*70}\n")

    # 加载时间序列数据
    print("Step 1: 加载时间序列数据...")
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)

    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    observed = {"A_t": observed_wound_area}
    
    print(f"  位置: {ts_data['location']}")
    print(f"  时间点: {observed_time}")
    print(f"  伤口面积: {observed_wound_area}")

    # 加载初始mask
    print("\nStep 2: 加载初始mask...")
    import pandas as pd
    from ca.model import CAParams
    
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

    # 使用优化参数
    if use_best_params:
        print("\nStep 3: 使用已优化的参数...")
        
        # 尝试加载已有的优化结果
        result_file = Path(__file__).parent / "results" / "results_L12.json"
        print(f"  查找优化结果文件: {result_file}")
        print(f"  文件存在: {result_file.exists()}")
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                saved_results = json.load(f)
            
            # 使用已保存的参数
            best_params = CAParams(
                p_move=saved_results['best_params']['p_move'],
                p_div=saved_results['best_params']['p_div'],
                alpha=saved_results['best_params']['alpha'],
                beta=saved_results['best_params']['beta'],
                edge_bonus=saved_results['best_params']['edge_bonus'],
                k_time=1.0,  # 默认值
                gamma=0.0,   # 默认值
            )
            print(f"  使用已保存的优化参数:")
            print(f"    p_move: {best_params.p_move:.4f}")
            print(f"    p_div: {best_params.p_div:.4f}")
            print(f"    alpha: {best_params.alpha:.4f}")
            print(f"    beta: {best_params.beta:.4f}")
            print(f"    edge_bonus: {best_params.edge_bonus:.4f}")
        else:
            print(f"  未找到已保存的优化结果")
            print(f"  使用默认参数...")
            best_params = CAParams(
                p_move=0.5,
                p_div=0.05,
                alpha=1.0,
                beta=1.0,
                edge_bonus=2.0,
                k_time=1.0,
                gamma=0.0,
            )
    else:
        print("\nStep 3: 使用默认参数...")
        best_params = CAParams(
            p_move=0.5,
            p_div=0.05,
            alpha=1.0,
            beta=1.0,
            edge_bonus=2.0,
            k_time=1.0,
            gamma=0.0,
        )

    # 运行完整模拟
    print(f"\nStep 4: 运行完整模拟...")
    print(f"  参数:")
    print(f"    p_move={best_params.p_move:.4f}, p_div={best_params.p_div:.4f}")
    print(f"    alpha={best_params.alpha:.4f}, beta={best_params.beta:.4f}")
    print(f"    edge_bonus={best_params.edge_bonus:.4f}")
    print(f"    k_time={best_params.k_time:.4f}, gamma={best_params.gamma:.4f}")
    
    start_time = time.time()
    history = run_simulation(
        params=best_params,
        initial_mask=initial_mask,
        num_steps=50,
        downsample_k=4,
        seed=42,
        observed_time=observed_time,
    )
    elapsed = time.time() - start_time
    
    print(f"  模拟完成，耗时: {elapsed:.1f}秒")

    # 提取结果
    simulated_wound_area = np.array([s["wound_area"] for s in history])
    sim_norm = simulated_wound_area / simulated_wound_area[0]
    obs_norm = observed_wound_area / observed_wound_area[0]

    # 计算指标
    from scipy.interpolate import interp1d
    sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
    f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
    sim_at_obs_time = f_sim(observed_time)
    
    mse = np.mean((sim_at_obs_time - obs_norm)**2)
    mae = np.mean(np.abs(sim_at_obs_time - obs_norm))

    print(f"\nStep 5: 计算质量指标...")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    print(f"\n  观测vs模拟 (归一化):")
    for i, (t, obs, sim) in enumerate(zip(observed_time, obs_norm, sim_at_obs_time)):
        diff = (sim - obs) * 100
        print(f"    t={t:3.0f}: 观测={obs:.4f}, 模拟={sim:.4f}, 差异={diff:+.2f}%")

    # 可视化
    print(f"\nStep 6: 生成可视化...")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"  输出目录: {output_dir}")
    print(f"  目录存在: {output_dir.exists()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 伤口面积对比
    ax1 = axes[0, 0]
    ax1.plot(sim_time_real, sim_norm, 'b-', label='Simulated', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=3, marker='s', markersize=10)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized wound area')
    ax1.set_title(f'Wound Area: Simulated vs Observed (MSE={mse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 点对点比较
    ax2 = axes[0, 1]
    x_pos = np.arange(len(observed_time))
    width = 0.35
    ax2.bar(x_pos - width/2, obs_norm, width, label='Observed', color='r', alpha=0.7)
    ax2.bar(x_pos + width/2, sim_at_obs_time, width, label='Simulated', color='b', alpha=0.7)
    ax2.set_xlabel('Time point')
    ax2.set_ylabel('Normalized wound area')
    ax2.set_title('Point-by-Point Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f't={t}' for t in observed_time])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3: 残差分析
    ax3 = axes[1, 0]
    residuals = (sim_at_obs_time - obs_norm) * 100
    colors = ['green' if abs(r) < 10 else 'orange' if abs(r) < 20 else 'red' for r in residuals]
    ax3.bar(x_pos, residuals, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.axhline(y=10, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax3.axhline(y=-10, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Time point')
    ax3.set_ylabel('Residual (%)')
    ax3.set_title('Residuals at Observed Time Points')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f't={t}' for t in observed_time])
    ax3.grid(True, alpha=0.3)

    # 子图4: 细胞活动
    ax4 = axes[1, 1]
    migrations = [s["migrations"] for s in history]
    divisions = [s["divisions"] for s in history]
    ax4.plot(migrations, 'b-', label='Migrations', linewidth=2, alpha=0.7)
    ax4.plot(divisions, 'r-', label='Divisions', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Simulation step')
    ax4.set_ylabel('Count')
    ax4.set_title('Cellular Activities Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / f"optimized_simulation_{location}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {plot_path}")
    plt.close()

    # 保存结果
    results = {
        "location": location,
        "simulation_time": elapsed,
        "params": {
            "p_move": float(best_params.p_move),
            "p_div": float(best_params.p_div),
            "alpha": float(best_params.alpha),
            "beta": float(best_params.beta),
            "edge_bonus": float(best_params.edge_bonus),
            "k_time": float(best_params.k_time),
            "gamma": float(best_params.gamma),
        },
        "metrics": {
            "mse": float(mse),
            "mae": float(mae),
        },
        "observed_time": observed_time.tolist(),
        "observed_wound_area_norm": obs_norm.tolist(),
        "simulated_wound_area_norm": sim_norm.tolist(),
        "simulated_at_obs_time": sim_at_obs_time.tolist(),
        "plot_path": str(plot_path),
    }

    results_path = output_dir / f"optimized_simulation_{location}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  保存结果: {results_path}")

    print(f"\n{'='*70}")
    print("模拟完成！")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    # 使用优化参数运行模拟
    results = run_optimized_simulation(
        location="L12",
        use_best_params=True,
    )

    if results:
        print("\n=== 最终结果 ===")
        print(f"位置: {results['location']}")
        print(f"模拟时间: {results['simulation_time']:.1f}秒")
        print(f"\n使用的参数:")
        for k, v in results['params'].items():
            print(f"  {k}: {v:.4f}")
        print(f"\n质量指标:")
        print(f"  MSE: {results['metrics']['mse']:.6f}")
        print(f"  MAE: {results['metrics']['mae']:.6f}")
