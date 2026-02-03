"""
运行贝叶斯优化以获得更准确的CA模型参数
"""
import numpy as np
import json
import sys
from pathlib import Path

from core.utils.io_mat import load_binary_mask
from core.inference.fit_bo import fit_bayesian_optimization, run_simulation
from core.ca.model import CAParams
import matplotlib.pyplot as plt

def run_bayesian_optimization_example(
    location="L12",
    n_calls=50,
    n_initial_points=10,
    downsample_k=4
):
    """
    运行贝叶斯优化以找到最佳CA参数

    Args:
        location: 位置标识符
        n_calls: 总优化迭代次数
        n_initial_points: 初始随机采样点数
        downsample_k: 降采样因子
    """
    print(f"\n{'='*70}")
    print(f"贝叶斯优化 - 位置 {location}")
    print(f"{'='*70}\n")

    # 加载时间序列数据
    print("Step 1: 加载时间序列数据...")
    ts_file = f"time_series_{location}.json"
    
    if not Path(ts_file).exists():
        print(f"错误: 时间序列文件不存在: {ts_file}")
        return None

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
    first_frame_id = ts_data["frame_ids"][0]
    
    # 从manifest加载mask路径
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    row = manifest[manifest["frame_id"] == first_frame_id]
    
    if len(row) == 0:
        print(f"错误: 在manifest中找不到frame_id {first_frame_id}")
        return None

    mask_path = row["manual_mask_path"].values[0]
    
    # Convert relative path to absolute path if needed
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        # Path is relative, make it absolute from project root
        project_root = Path(__file__).parent.parent
        mask_path = str(project_root / mask_path)
    
    initial_mask = load_binary_mask(mask_path)
    
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Cell coverage: {np.mean(initial_mask):.3f}")

    # 运行贝叶斯优化
    print(f"\nStep 3: 运行贝叶斯优化...")
    print(f"  总迭代次数: {n_calls}")
    print(f"  初始随机采样: {n_initial_points}")
    print(f"  降采样因子: {downsample_k}")

    result = fit_bayesian_optimization(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        n_iter=n_initial_points,
        n_calls=n_calls,
        random_state=42,
        downsample_k=downsample_k,
    )

    print(f"\n  最优损失: {result['best_loss']:.6f}")
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
    mse = np.mean((sim_norm - obs_norm)**2)
    mae = np.mean(np.abs(sim_norm - obs_norm))

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
    sim_time = np.linspace(0, observed_time[-1], len(sim_norm))
    ax1.plot(sim_time, sim_norm, 'b-', label='Simulated', linewidth=2, marker='o', markersize=3)
    ax1.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=2, marker='s', markersize=8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized wound area')
    ax1.set_title(f'Wound Area Comparison (MSE={mse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 优化历史
    ax2 = axes[0, 1]
    if hasattr(result['result'], 'func_vals'):
        func_vals = result['result'].func_vals
        ax2.plot(range(1, len(func_vals) + 1), func_vals, 'g-', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Optimization History')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    # 子图3: 参数分布（如果有）
    ax3 = axes[1, 0]
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
    ax3.barh(param_names, param_values, color=colors, alpha=0.7)
    ax3.set_xlabel('Parameter value')
    ax3.set_title('Optimized Parameters')
    ax3.grid(True, alpha=0.3, axis='x')

    # 子图4: 残差分析
    ax4 = axes[1, 1]
    # 插值模拟数据到观测时间点
    from scipy.interpolate import interp1d
    f_sim = interp1d(sim_time, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
    sim_at_obs = f_sim(observed_time)
    residuals = (sim_at_obs - obs_norm) * 100  # 百分比残差
    ax4.bar(range(len(observed_time)), residuals, color='orange', alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax4.set_xlabel('Time point index')
    ax4.set_ylabel('Residual (%)')
    ax4.set_title('Residuals at Observed Time Points')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / f"bayesian_opt_{location}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {plot_path}")
    plt.close()

    # 保存结果
    results = {
        "location": location,
        "optimization_method": "Bayesian Optimization",
        "n_calls": n_calls,
        "n_initial_points": n_initial_points,
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

    results_path = output_dir / f"bayesian_opt_{location}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  保存结果: {results_path}")

    print(f"\n{'='*70}")
    print("贝叶斯优化完成！")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    # 运行贝叶斯优化（20次迭代，用于快速测试）
    results = run_bayesian_optimization_example(
        location="L12",
        n_calls=20,           # 总迭代次数（减少以加快速度）
        n_initial_points=5,   # 初始随机采样
        downsample_k=4,
    )

    if results:
        print("\n=== 最终结果 ===")
        print(f"位置: {results['location']}")
        print(f"最优参数:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v:.4f}")
        print(f"\n质量指标:")
        print(f"  MSE: {results['metrics']['mse']:.6f}")
        print(f"  MAE: {results['metrics']['mae']:.6f}")
