import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from pathlib import Path
from io_mat import load_binary_mask
from preprocess.extract_observations import downsample_binary
from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import run_simulation, calculate_loss
import pandas as pd
import time

# Simple test without plotting
print("=" * 60)
print("Simple CA Test (no fitting, no plotting)")
print("=" * 60)

# Load data
print("\n1. Loading data...")
start = time.time()
with open("CA_project/time_series_L12.json", 'r') as f:
    ts_data = json.load(f)
print(f"   Time series loaded: {len(ts_data['frame_ids'])} frames")
print(f"   Time points: {ts_data['time_points']}")
print(f"   Wound areas: {ts_data['wound_area']}")

manifest = pd.read_csv('CA_project/data_manifest.csv')
first_frame_id = ts_data['frame_ids'][0]
row = manifest[manifest['frame_id'] == first_frame_id]
mask = load_binary_mask(row['manual_mask_path'].values[0])
print(f"   Mask loaded: {mask.shape}")
print(f"   Took {time.time() - start:.2f}s")

# Downsample
print("\n2. Downsampling...")
mask_ds = downsample_binary(mask, k=8, thr=0.5)
print(f"   Downsampled to: {mask_ds.shape}")

# Create CA
print("\n3. Creating CA model...")
H, W = mask_ds.shape
params = CAParams(p_move=0.8, p_div=0.1, alpha=1.0, beta=1.0, edge_bonus=2.0)
ca = CellOnlyCA(H, W, params)
ca.grid = mask_ds.astype(np.int32)
print(f"   CA grid: {H}x{W}")
print(f"   Initial wound area: {np.sum(ca.grid == 0)}")

# Run simulation
print("\n4. Running simulation...")
num_steps = min(int(ts_data['time_points'][-1]) + 1, 50)
print(f"   Running {num_steps} steps...")
start = time.time()

history = []
for step in range(num_steps):
    stats = ca.step()
    stats['wound_area'] = float(np.sum(ca.grid == 0))
    stats['step'] = step
    history.append(stats)
    
    if step % 10 == 0:
        print(f"   Step {step}/{num_steps}: wound_area={stats['wound_area']}")

elapsed = time.time() - start
print(f"   Simulation took {elapsed:.2f}s ({elapsed/num_steps*1000:.1f} ms/step)")

# Calculate loss
print("\n5. Calculating loss...")
observed_wound_area = np.array(ts_data['wound_area'])
observed_time = np.array(ts_data['time_points'])
simulated_wound_area = np.array([s['wound_area'] for s in history])

print(f"   Observed wound area (raw): {observed_wound_area}")
print(f"   Simulated wound area (raw): {simulated_wound_area[[0, len(simulated_wound_area)//2, -1]]}")

# Normalize
obs_norm = observed_wound_area / observed_wound_area[0]
sim_norm = simulated_wound_area / simulated_wound_area[0]

print(f"   Observed (normalized): {obs_norm}")
print(f"   Simulated (normalized): {sim_norm[[0, len(sim_norm)//2, -1]]}")

# Calculate MSE on normalized data at observed time points
from scipy.interpolate import interp1d
sim_time = np.arange(len(simulated_wound_area))
f_sim = interp1d(sim_time, sim_norm, kind='linear', fill_value='extrapolate')
sim_at_obs_time = f_sim(observed_time)

mse = np.mean((sim_at_obs_time - obs_norm) ** 2)
print(f"   MSE (normalized, at observed times): {mse:.4f}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
