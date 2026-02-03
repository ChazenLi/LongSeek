"""
Quick optimization test with gamma and k_time

Tests if including gamma and k_time in optimization improves results.
"""

import numpy as np
import json
import sys
from pathlib import Path

from core.utils.io_mat import load_binary_mask
from core.ca.model import CAParams
from core.inference.fit_bo import fit_simple_grid_search, run_simulation, calculate_loss

def quick_optimization_with_gamma_ktime(
    location="L12",
    n_samples=30,  # Reduced for speed
    use_gamma_ktime=True
):
    """Quick optimization test."""
    print(f"\n{'='*70}")
    print(f"Quick Optimization Test - Location {location}")
    print(f"Include gamma/k_time: {use_gamma_ktime}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Step 1: Loading data...")
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)
    
    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    obs_norm = observed_wound_area / observed_wound_area[0]
    
    # Load initial mask
    print("Step 2: Loading initial mask...")
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    first_frame_id = ts_data["frame_ids"][0]
    row = manifest[manifest["frame_id"] == first_frame_id]
    mask_path = row["manual_mask_path"].values[0]
    
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        mask_path = str(Path(__file__).parent.parent / mask_path)
    
    initial_mask = load_binary_mask(mask_path)
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Initial wound area: {observed_wound_area[0]:.0f}")
    print(f"  Final wound area: {observed_wound_area[-1]:.0f}")
    
    # Prepare observed data
    observed = {
        "A_t": observed_wound_area,
    }
    
    # Run optimization
    print(f"\nStep 3: Running optimization ({n_samples} samples)...")
    
    if use_gamma_ktime:
        print("  Search space: p_move, p_div, alpha, beta, edge_bonus, k_time, gamma")
        result = fit_simple_grid_search(
            initial_mask=initial_mask,
            observed=observed,
            observed_time=observed_time,
            downsample_k=4,
            n_samples=n_samples,
        )
    else:
        print("  Search space: p_move, p_div, alpha, beta, edge_bonus (k_time=1.0, gamma=0.0)")
        result = fit_simple_grid_search(
            initial_mask=initial_mask,
            observed=observed,
            observed_time=observed_time,
            downsample_k=4,
            n_samples=n_samples,
        )
    
    best_params = result["best_params"]
    best_loss = result["best_loss"]
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss (MSE): {best_loss:.6f}")
    print(f"\nBest parameters:")
    print(f"  p_move:     {best_params.p_move:.6f}")
    print(f"  p_div:      {best_params.p_div:.6f}")
    print(f"  alpha:      {best_params.alpha:.6f}")
    print(f"  beta:       {best_params.beta:.6f}")
    print(f"  edge_bonus: {best_params.edge_bonus:.6f}")
    print(f"  k_time:     {best_params.k_time:.6f}")
    print(f"  gamma:      {best_params.gamma:.6f}")
    print(f"{'='*70}\n")
    
    # Evaluate best parameters
    print("Step 4: Evaluating best parameters...")
    history = run_simulation(
        params=best_params,
        initial_mask=initial_mask,
        num_steps=50,
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
    t17_error = (sim_at_obs[1] - obs_norm[1]) * 100
    t40_error = (sim_at_obs[2] - obs_norm[2]) * 100
    
    print(f"\nEvaluation:")
    print(f"  MSE:          {mse:.6f}")
    print(f"  t=17 error:   {t17_error:+.2f}%")
    print(f"  t=40 error:   {t40_error:+.2f}%")
    print(f"\n  Observed vs Simulated at t=40:")
    print(f"    Observed: {obs_norm[2]:.4f}")
    print(f"    Simulated: {sim_at_obs[2]:.4f}")
    
    # Compare with baseline (optimized without gamma/k_time)
    baseline_mse = 0.017
    baseline_t40_error = 61.8
    
    mse_improvement = (baseline_mse - mse) / baseline_mse * 100
    t40_improvement = (baseline_t40_error - t40_error) / baseline_t40_error * 100
    
    print(f"\nComparison with baseline:")
    print(f"  MSE improvement:    {mse_improvement:+.1f}%")
    print(f"  t=40 improvement:  {t40_improvement:+.1f}%")
    
    # Check success criteria
    success = t40_error < 40  # Target from ROADMAP
    print(f"\nSuccess criteria (t=40 < 40%): {'PASS' if success else 'FAIL'}")
    
    # Save results
    results = {
        'location': location,
        'use_gamma_ktime': use_gamma_ktime,
        'best_params': {
            'p_move': float(best_params.p_move),
            'p_div': float(best_params.p_div),
            'alpha': float(best_params.alpha),
            'beta': float(best_params.beta),
            'edge_bonus': float(best_params.edge_bonus),
            'k_time': float(best_params.k_time),
            'gamma': float(best_params.gamma),
        },
        'best_loss': float(best_loss),
        'evaluation': {
            'mse': float(mse),
            't17_error': float(t17_error),
            't40_error': float(t40_error),
        },
        'comparison': {
            'baseline_mse': baseline_mse,
            'baseline_t40_error': baseline_t40_error,
            'mse_improvement_pct': mse_improvement,
            't40_improvement_pct': t40_improvement,
        },
        'success': success,
        'observed_time': observed_time.tolist(),
        'observed_wound_area_norm': obs_norm.tolist(),
        'simulated_at_obs_time': sim_at_obs.tolist(),
    }
    
    output_path = Path(f"results/quick_optimization_gamma_ktime_{use_gamma_ktime}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {output_path}")
    
    return results

if __name__ == "__main__":
    # Test with gamma and k_time
    results_with = quick_optimization_with_gamma_ktime(location="L12", n_samples=30, use_gamma_ktime=True)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results_with['success']:
        print("✓ Optimization with gamma/k_time achieved t=40 error < 40%")
        print(f"  Best t=40 error: {results_with['evaluation']['t40_error']:.2f}%")
        print(f"  Best gamma: {results_with['best_params']['gamma']:.3f}")
        print(f"  Best k_time: {results_with['best_params']['k_time']:.3f}")
    else:
        print("✗ Optimization did not achieve t=40 error < 40%")
        print(f"  Best t=40 error: {results_with['evaluation']['t40_error']:.2f}%")
        print(f"  Consider: more samples, different model, or additional observables")
