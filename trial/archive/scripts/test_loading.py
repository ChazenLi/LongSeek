"""
测试数据加载和基本CA模拟
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

def test_data_loading():
    """测试数据加载"""
    print("Step 1: 测试数据加载...")
    
    # 加载时间序列
    with open("time_series_L12.json", 'r') as f:
        ts_data = json.load(f)
    
    print(f"  时间点: {ts_data['time_points']}")
    print(f"  伤口面积: {ts_data['wound_area']}")
    
    # 加载manifest
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    first_frame_id = ts_data["frame_ids"][0]
    print(f"  第一帧ID: {first_frame_id}")
    
    row = manifest[manifest["frame_id"] == first_frame_id]
    mask_path = row["manual_mask_path"].values[0]
    print(f"  Mask路径: {mask_path}")
    
    # 转换为绝对路径
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        project_root = Path(__file__).parent.parent
        mask_path = str(project_root / mask_path)
    print(f"  绝对路径: {mask_path}")
    print(f"  文件存在: {Path(mask_path).exists()}")
    
    # 加载mask
    print("\nStep 2: 加载mask...")
    start = time.time()
    initial_mask = load_binary_mask(mask_path)
    elapsed = time.time() - start
    print(f"  加载时间: {elapsed:.2f}秒")
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Cell coverage: {np.mean(initial_mask):.3f}")
    
    # 降采样
    print("\nStep 3: 降采样...")
    start = time.time()
    mask_ds = downsample_binary(initial_mask, k=4, thr=0.5)
    elapsed = time.time() - start
    print(f"  降采样时间: {elapsed:.2f}秒")
    print(f"  降采样shape: {mask_ds.shape}")
    
    # CA模拟
    print("\nStep 4: 运行CA模拟...")
    params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, edge_bonus=2.0)
    
    ca = CellOnlyCA(height=mask_ds.shape[0], width=mask_ds.shape[1], params=params)
    ca.initialize_from_mask(initial_mask, k=4)
    
    # 运行几步
    num_steps = 10
    print(f"  运行 {num_steps} 步...")
    start = time.time()
    history = ca.run(num_steps=num_steps)
    elapsed = time.time() - start
    
    print(f"  模拟时间: {elapsed:.2f}秒 ({elapsed/num_steps*1000:.1f}ms/步)")
    print(f"  初始伤口面积: {history[0]['wound_area']}")
    print(f"  最终伤口面积: {history[-1]['wound_area']}")
    
    print("\n✓ 所有测试通过！")
    return True

if __name__ == "__main__":
    try:
        test_data_loading()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
