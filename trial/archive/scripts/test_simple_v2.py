"""
Simple test to verify the CA model works correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
import time

# Import with correct paths
from io_mat import load_binary_mask
from preprocess.extract_observations import downsample_binary
from ca.model import CellOnlyCA, CAParams
import pandas as pd

# Simple test without fitting or plotting
print("=" * 60)
print("Simple CA Test")
print("=" * 60)

# Load data
print("\n1. Loading data...")
start = time.time()

with open("time_series_L12.json", 'r') as f:
    ts_data = json.load(f)

print(f"   Time series: {len(ts_data['frame_ids'])} frames")
print(f"   Time points: {ts_data['time_points']}")
print(f"   Wound areas: {ts_data['wound_area']}")

manifest = pd.read_csv('data_manifest.csv')
first_frame_id = ts_data['frame_ids'][0]
row = manifest[manifest['frame_id'] == first_frame_id]
mask = load_binary_mask(row['manual_mask_path'].values[0])

print(f"   Mask shape: {mask.shape}")
print(f"   Loading took {time.time() - start:.2f}s")

# Downsample
print("\n2. Downsampling...")
mask_ds = downsample_binary(mask, k=8, thr=0.5)
print(f"   Downsampled to: {mask_ds.shape}")
print(f"   Cell coverage: {mask_ds.mean():.3f}")

# Create CA
print("\n3. Creating CA model...")
H, W = mask_ds.shape
params = CAParams(p_move=0.8, p_div=0.1, alpha=1.0, beta=1.0, edge_bonus=2.0)
ca = CellOnlyCA(H, W, params)
ca.grid = mask_ds.astype(np.int32)
initial_wound = np.sum(ca.grid == 0)
print(f"   CA grid: {H}x{W}")
print(f"   Initial wound area: {initial_wound}")

# Run simulation
print("\n4. Running simulation...")
num_steps = 41
start = time.time()

wound_areas = []
for step in range(num_steps):
    stats = ca.step()
    wound_area = np.sum(ca.grid == 0)
    wound_areas.append(wound_area)
    
    if step % 10 == 0 or step == num_steps - 1:
        print(f"   Step {step:2d}/{num_steps}: wound_area={wound_area:5d} ({wound_area/initial_wound*100:5.1f}%)")

elapsed = time.time() - start
print(f"\n   Simulation took {elapsed:.2f}s ({elapsed/num_steps*1000:.1f} ms/step)")

# Compare with observed
print("\n5. Comparing with observed data...")
observed_wound_area = np.array(ts_data['wound_area'])
observed_time = np.array(ts_data['time_points'])

# Normalize both
obs_norm = observed_wound_area / observed_wound_area[0]
sim_wound_arr = np.array(wound_areas)
sim_norm = sim_wound_arr / sim_wound_arr[0]

print(f"\n   Observed time points: {observed_time}")
print(f"   Observed (normalized): {obs_norm}")
print(f"\n   Simulated at observed times:")

# Interpolate to observed times
from scipy.interpolate import interp1d
sim_time = np.arange(len(sim_wound_arr))
f_sim = interp1d(sim_time, sim_norm, kind='linear', fill_value='extrapolate')
sim_at_obs = f_sim(observed_time)

for i, t in enumerate(observed_time):
    print(f"     t={t:2d}: {sim_at_obs[i]:.4f} (observed: {obs_norm[i]:.4f}, diff: {abs(sim_at_obs[i] - obs_norm[i]):.4f})")

mse = np.mean((sim_at_obs - obs_norm) ** 2)
print(f"\n   MSE: {mse:.4f}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
