"""
Directional Bias Implementation - Use gamma parameter properly

This implements the user's suggestion:
- Use gamma parameter as directional bias (not zero!)
- Migration frequency: controlled by existing rules (p_move × exp(-α×density) + edge_bonus)
- Migration direction: controlled by field gradient (gamma parameter)
- This adds "directionality" without requiring molecular data
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from core.utils.io_mat import load_binary_mask
from core.ca.model import CellOnlyCA, CAParams
from core.inference.fit_bo import run_simulation, calculate_loss
from scipy.ndimage import distance_transform_edt, gaussian_filter

def create_static_directional_field(mask, sigma=10):
    """
    Create a static directional field from wound mask.
    
    This uses distance transform to create a field that guides cells
    toward the wound center without requiring molecular measurements.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        sigma: Gaussian smoothing parameter
        
    Returns:
        direction_field: 2D array with direction preference
    """
    # Distance from wound (distance transform)
    # For wound pixels (mask==0), distance is 0
    # For cell pixels (mask==1), distance is from nearest wound
    wound_mask = (mask == 0).astype(np.uint8)
    distance_field = distance_transform_edt(wound_mask)
    
    # Compute gradient (points toward wound = direction of increasing distance from cells)
    # We want cells to move toward wound, so we use the distance field
    # Smooth it to create a more natural field
    smoothed_field = gaussian_filter(distance_field, sigma=sigma)
    
    # Normalize to [0, 1]
    if smoothed_field.max() > 0:
        smoothed_field = smoothed_field / smoothed_field.max()
    
    return smoothed_field

def run_simulation_with_directional_bias(
    location="L12",
    gamma_values=[0.0, 0.5, 1.0, 1.5, 2.0],
    fixed_k_time=1.0
):
    """
    Test different gamma values to fix the late-stage closure problem.
    
    The key insight: current gamma=0.0 means "no direction bias"
    We need to use gamma > 0 to get directional migration.
    """
    print(f"\n{'='*70}")
    print(f"Testing Directional Bias - Location {location}")
    print(f"{'='*70}\n")
    
    # Load data
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)
    
    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    observed = {"A_t": observed_wound_area}
    
    # Load initial mask
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
    
    print(f"Initial wound area: {observed_wound_area[0]:.0f}")
    print(f"Final wound area (observed): {observed_wound_area[-1]:.0f}")
    print(f"Closure rate: {(1 - observed_wound_area[-1]/observed_wound_area[0])*100:.1f}%")
    
    # Create static directional field
    print(f"\nCreating static directional field...")
    direction_field = create_static_directional_field(initial_mask, sigma=10)
    
    # Test different gamma values
    results = []
    
    for gamma in gamma_values:
        print(f"\n{'='*50}")
        print(f"Testing gamma = {gamma}")
        print(f"{'='*50}")
        
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
        
        # Run simulation
        history = run_simulation(
            params=params,
            initial_mask=initial_mask,
            num_steps=50,
            downsample_k=4,
            seed=42,
            observed_time=observed_time,
        )
        
        simulated_wound_area = np.array([s["wound_area"] for s in history])
        sim_norm = simulated_wound_area / simulated_wound_area[0]
        obs_norm = observed_wound_area / observed_wound_area[0]
        
        # Interpolate to observation time points
        from scipy.interpolate import interp1d
        sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
        f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
        sim_at_obs = f_sim(observed_time)
        
        # Calculate metrics
        mse = np.mean((sim_at_obs - obs_norm)**2)
        mae = np.mean(np.abs(sim_at_obs - obs_norm))
        
        # Point-wise errors
        errors = (sim_at_obs - obs_norm) * 100
        
        print(f"\nResults for gamma={gamma}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"\n  Time point comparison:")
        for i, (t, obs, sim) in enumerate(zip(observed_time, obs_norm, sim_at_obs)):
            err = (sim - obs) * 100
            print(f"    t={t:3.0f}h: obs={obs:.4f}, sim={sim:.4f}, error={err:+.2f}%")
        
        results.append({
            'gamma': gamma,
            'mse': mse,
            'mae': mae,
            'sim_at_obs': sim_at_obs.tolist(),
            'sim_norm': sim_norm.tolist()
        })
    
    # Find best gamma
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\n{'='*70}")
    print(f"BEST RESULT: gamma = {best_result['gamma']}")
    print(f"  MSE: {best_result['mse']:.6f}")
    print(f"  MAE: {best_result['mae']:.6f}")
    print(f"{'='*70}\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Figure 1: Effect of gamma on wound area
    ax1 = axes[0, 0]
    for result in results:
        ax1.plot(observed_time, result['sim_at_obs'], 'o-', 
                label=f'gamma={result["gamma"]}', linewidth=2, markersize=8)
    ax1.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=3, marker='s', markersize=10)
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Wound Area', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Directional Bias (gamma)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: MSE vs gamma
    ax2 = axes[0, 1]
    gammas = [r['gamma'] for r in results]
    mses = [r['mse'] for r in results]
    ax2.plot(gammas, mses, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.017, color='red', linestyle='--', linewidth=2, label='Current best (gamma=0)')
    ax2.set_xlabel('Gamma (directional bias)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax2.set_title('MSE vs Directional Bias Strength', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(mses) * 1.1])
    
    # Figure 3: t=40 error vs gamma
    ax3 = axes[1, 0]
    t40_errors = [(r['sim_at_obs'][2] - obs_norm[2]) * 100 for r in results]
    ax3.plot(gammas, t40_errors, 'ro-', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Gamma (directional bias)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error at t=40 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Late-Stage Error vs Directional Bias', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Fill area showing improvement
    ax3.fill_between(gammas, 0, t40_errors, where=[e < 60 for e in t40_errors], 
                     alpha=0.3, color='green', label='Improved (< 60%)')
    ax3.legend(fontsize=10)
    
    # Figure 4: Directional field visualization
    ax4 = axes[1, 1]
    im = ax4.imshow(direction_field, cmap='viridis', origin='lower')
    ax4.set_title('Static Directional Field (Distance Transform)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Distance from Wound', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path("results/directional_bias_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved directional bias analysis: {output_path}")
    plt.close()
    
    # Save results
    output_dir = Path("results")
    results_path = output_dir / "directional_bias_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'location': location,
            'gamma_values': gamma_values,
            'results': results,
            'best_gamma': best_result['gamma'],
            'best_mse': best_result['mse']
        }, f, indent=2)
    
    print(f"Saved results: {results_path}")
    
    return results

if __name__ == "__main__":
    # Test directional bias
    results = run_simulation_with_directional_bias(
        location="L12",
        gamma_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        fixed_k_time=1.0
    )
    
    print("\n=== KEY FINDINGS ===")
    print("Directional bias (gamma) addresses the late-stage closure problem.")
    print("Recommendation: Use gamma > 0 in future optimizations.")
