"""
快速真实数据测试 - 单个数据集
"""
import numpy as np
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask
from preprocess.extract_observations import extract_frame_statistics, downsample_binary
from ca.model import CellOnlyCA, CAParams

def quick_real_data_test(dataset_name="SN15", num_frames=3, num_steps=20):
    """快速测试真实数据"""

    print(f"\n{'='*70}")
    print(f"快速测试: {dataset_name} 数据集")
    print(f"{'='*70}\n")

    # 路径
    project_root = Path(__file__).parent.parent
    base_path = project_root / "CA" / "DATA" / dataset_name / dataset_name
    manual_dir = base_path / "manual_mat"

    print(f"数据路径: {base_path}")
    print(f"Mask目录: {manual_dir}")
    print(f"存在: {manual_dir.exists()}\n")

    if not manual_dir.exists():
        print("错误: 数据目录不存在")
        return

    # 获取mask文件
    mat_files = sorted([f for f in manual_dir.glob("*.mat") if not f.name.startswith("desktop")])
    print(f"找到 {len(mat_files)} 个mask文件\n")

    if len(mat_files) == 0:
        print("错误: 没有找到mask文件")
        return

    # 加载前几个mask
    print(f"加载前 {min(num_frames, len(mat_files))} 个mask:")
    masks = []
    observations = []

    for i, mat_file in enumerate(mat_files[:num_frames]):
        try:
            print(f"  [{i+1}] {mat_file.name}")
            mask = load_binary_mask(str(mat_file))
            stats = extract_frame_statistics(mask)

            masks.append(mask)
            observations.append(stats)

            print(f"      shape={mask.shape}, "
                  f"wound_area={stats['wound_area']:.0f}, "
                  f"coverage={stats['cell_coverage']:.3f}")
        except Exception as e:
            print(f"      错误: {e}")

    if len(masks) == 0:
        print("\n错误: 没有成功加载任何mask")
        return

    # 使用第一个mask作为初始状态
    print(f"\n使用第一个mask运行CA模拟...")
    initial_mask = masks[0]

    # 降采样
    downsample_k = 8  # 更大的降采样以提高速度
    H, W = initial_mask.shape
    H_ds, W_ds = H // downsample_k, W // downsample_k
    print(f"  降采样: {initial_mask.shape} -> ({H_ds}, {W_ds})")

    # CA参数
    params = CAParams(
        p_move=0.5,
        p_div=0.05,
        alpha=1.0,
        beta=1.0,
        edge_bonus=2.0
    )
    print(f"  参数: p_move={params.p_move}, p_div={params.p_div}\n")

    # 运行CA
    print(f"运行 {num_steps} 步模拟...")
    start = time.time()

    ca = CellOnlyCA(height=H_ds, width=W_ds, params=params)
    ca.initialize_from_mask(initial_mask, k=downsample_k)
    history = ca.run(num_steps=num_steps)

    elapsed = time.time() - start
    print(f"完成: {elapsed:.1f}秒 ({elapsed/num_steps*1000:.1f}ms/步)\n")

    # 结果
    sim_wound_area = [step["wound_area"] for step in history]
    obs_wound_area = [obs["wound_area"] for obs in observations]

    # 归一化
    sim_norm = np.array(sim_wound_area) / sim_wound_area[0]
    obs_norm = np.array(obs_wound_area) / obs_wound_area[0]

    print("结果:")
    print(f"  观测伤口面积: {obs_wound_area}")
    print(f"  模拟伤口面积: {sim_wound_area[:5]}... (共{len(sim_wound_area)}步)")

    print(f"\n  归一化伤口面积:")
    print(f"    观测: {obs_norm}")
    print(f"    模拟: {sim_norm[:5]}...")

    print(f"\n  伤口闭合率:")
    print(f"    观测: {(1-obs_norm[-1])*100:.1f}%")
    print(f"    模拟: {(1-sim_norm[-1])*100:.1f}%")

    # 保存简单结果
    results = {
        "dataset": dataset_name,
        "num_frames": len(masks),
        "num_steps": num_steps,
        "observed_wound_area": [float(x) for x in obs_wound_area],
        "simulated_wound_area": [float(x) for x in sim_wound_area],
        "observed_closure_rate": float((1-obs_norm[-1])*100),
        "simulated_closure_rate": float((1-sim_norm[-1])*100),
    }

    output_file = Path(__file__).parent / f"results_{dataset_name}_quick.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_file}")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    # 测试所有4个数据集
    datasets = ["SN15", "MDCK", "Melanoma", "TScratch"]

    all_results = {}
    for dataset in datasets:
        try:
            result = quick_real_data_test(
                dataset_name=dataset,
                num_frames=3,   # 每个数据集3帧
                num_steps=20    # 模拟20步
            )
            all_results[dataset] = result
        except Exception as e:
            print(f"错误: {dataset} - {e}\n")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    for dataset, result in all_results.items():
        print(f"\n{dataset}:")
        print(f"  观测闭合率: {result['observed_closure_rate']:.1f}%")
        print(f"  模拟闭合率: {result['simulated_closure_rate']:.1f}%")
        print(f"  差异: {abs(result['observed_closure_rate'] - result['simulated_closure_rate']):.1f}%")
