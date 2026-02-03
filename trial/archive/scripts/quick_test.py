"""
Quick test of CA model functionality.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ca.model import CellOnlyCA, CAParams
import time

print("=" * 60)
print("Quick CA Model Test")
print("=" * 60)

# Create a simple initial mask
print("\n1. Creating initial mask...")
mask = np.ones((64, 64), dtype=np.int32)
mask[16:48, 16:48] = 0  # Wound in center
print(f"   Mask shape: {mask.shape}")
print(f"   Cell coverage: {np.mean(mask):.3f}")
print(f"   Wound area: {np.sum(mask == 0)}")

# Create CA with default parameters
print("\n2. Creating CA model...")
params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, edge_bonus=2.0)
print(f"   Parameters:")
print(f"     p_move: {params.p_move}")
print(f"     p_div: {params.p_div}")
print(f"     alpha: {params.alpha}")
print(f"     beta: {params.beta}")
print(f"     edge_bonus: {params.edge_bonus}")

ca = CellOnlyCA(height=64, width=64, params=params)
ca.initialize_from_mask(mask, k=1)
print(f"   CA grid initialized: {ca.grid.shape}")

# Run simulation
print("\n3. Running simulation (20 steps)...")
start_time = time.time()
history = ca.run(num_steps=20)
elapsed = time.time() - start_time

print(f"   Completed in {elapsed:.2f}s ({elapsed/20*1000:.1f}ms/step)")

# Show results
print("\n4. Results:")
for i, step in enumerate([0, 10, 19]):
    if i < len(history):
        s = history[step]
        print(f"   Step {step}: wound_area={s['wound_area']}, "
              f"migrations={s['migrations']}, divisions={s['divisions']}")

print("\n5. Wound closure:")
initial_area = history[0]['wound_area']
final_area = history[-1]['wound_area']
closure_rate = (initial_area - final_area) / initial_area * 100
print(f"   Initial: {initial_area}")
print(f"   Final: {final_area}")
print(f"   Closure: {closure_rate:.1f}%")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
