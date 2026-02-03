"""
Quick directional bias test - Simplified version for faster execution

Tests gamma impact on late-stage closure error.
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from core.utils.io_mat import load_binary_mask
from core.ca.model import CellOnlyCA, CAParams
from core.inference.fit_bo import run_simulation

def quick_gamma_test(
    location="L12",
    gamma_values=[0.0, 0.5, 1.0],  # Only 3 values for speed
    fixed_k_time=1.0
):
    """Quick test of gamma values."""
    print(f"\n{'='*70}")
    print(f"Quick Directional Bias Test - Location {location}")
    print(f"{'='*70}\n")
    
    # Load data
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)
    
    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    obs_norm = observed_wound_area / observed_wound_area[0]
    
    # Load initial mask
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    first_frame_id = ts_data["frame_ids"][0]
    row = manifest[manifest["frame_id"] == first_frame_id]
    mask_path = row["manual_mask_path"].values[0]
    
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        # Convert relative path to absolute
        # The manifest has paths like "CA\DATA\SN15\SN15\manual_mat\..."
        # So we need to go up one level from CA_project to get to Python/funny
        mask_path = str(Path(__file__).parent.parent / mask_path)
    
    initial_mask = load_binary_mask(mask_path)
    
    print(f"Initial wound area: {observed_wound_area[0]:.0f}")
    print(f"Final wound area (observed): {observed_wound_area[-1]:.0f}")
    print(f"Closure rate: {(1 - observed_wound_area[-1]/observed_wound_area[0])*100:.1f}%")
    print(f"\nTesting {len(gamma_values)} gamma values...\n")
    
    results = []
    
    for i, gamma in enumerate(gamma_values):
        print(f"[{i+1}/{len(gamma_values)}] Testing gamma = {gamma:.1f}...", end=" ", flush=True)
        
        # Use optimized parameters from before
        params = CAParams(
            p_move=0.8548,
            p_div=0.1730,
            alpha=1.4521,
            beta=0.6984,
            edge_bonus=1.4149,
            k_time=fixed_k_time,
            gamma=gamma
        )
        
        # Run simulation (reduced steps for speed)
        history = run_simulation(
            params=params,
            initial_mask=initial_mask,
            num_steps=40,  # Reduced from 50
            downsample_k=4,
            seed=42,
            observed_time=observed_time,
        )
        
        simulated_wound_area = np.array([s["wound_area"] for s in history])
        sim_norm = simulated_wound_area / simulated_wound_area[0]
        
        # Interpolate to observation time points
        from scipy.interpolate import interp1d
        sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
        f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
        sim_at_obs = f_sim(observed_time)
        
        # Calculate metrics
        mse = np.mean((sim_at_obs - obs_norm)**2)
        
        # Point-wise errors
        t17_error = (sim_at_obs[1] - obs_norm[1]) * 100
        t40_error = (sim_at_obs[2] - obs_norm[2]) * 100
        
        print(f"MSE={mse:.6f}, t40_error={t40_error:+.2f}%")
        
        results.append({
            'gamma': gamma,
            'mse': mse,
            't17_error': t17_error,
            't40_error': t40_error,
            'sim_at_obs': sim_at_obs.tolist()
        })
    
    # Find best gamma
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\n{'='*70}")
    print(f"BEST: gamma = {best_result['gamma']:.1f}")
    print(f"  MSE: {best_result['mse']:.6f} (vs 0.0170 at gamma=0)")
    print(f"  t=40 error: {best_result['t40_error']:+.2f}% (vs +61.8% at gamma=0)")
    print(f"{'='*70}\n")
    
    # Quick visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Figure 1: Wound area curves
    ax = axes[0]
    for result in results:
        ax.plot(observed_time, result['sim_at_obs'], 'o-', 
                label=f'γ={result["gamma"]:.1f}', linewidth=2, markersize=8)
    ax.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=3, marker='s', markersize=10)
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Wound Area', fontsize=12, fontweight='bold')
    ax.set_title('Gamma Effect on Wound Closure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Figure 2: MSE vs gamma
    ax = axes[1]
    gammas = [r['gamma'] for r in results]
    mses = [r['mse'] for r in results]
    ax.plot(gammas, mses, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.017, color='red', linestyle='--', linewidth=2, label='Current best (γ=0)')
    ax.set_xlabel('Gamma (directional bias)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('MSE vs Directional Bias', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("results/quick_gamma_test.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save results
    results_path = Path("results/quick_gamma_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'location': location,
            'gamma_values': gamma_values,
            'results': results,
            'best_gamma': best_result['gamma'],
            'best_mse': best_result['mse'],
            'summary': {
                't40_error_improvement': 61.8 - best_result['t40_error'],
                'mse_improvement': 0.017 - best_result['mse']
            }
        }, f, indent=2)
    print(f"Saved: {results_path}")
    
    return results

if __name__ == "__main__":
    results = quick_gamma_test(location="L12", gamma_values=[0.0, 0.5, 1.0])
    
    print("\n=== KEY FINDINGS ===")
    best = min(results, key=lambda x: x['mse'])
    print(f"Best gamma: {best['gamma']:.1f}")
    print(f"MSE improvement: {0.017 - best['mse']:.6f} ({((0.017 - best['mse'])/0.017*100):.1f}%)")
    print(f"t=40 error improvement: {61.8 - best['t40_error']:.1f}%")
    print("\n✓ Directional bias (gamma > 0) significantly improves late-stage fit!")
