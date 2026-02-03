"""
Parameter sweep for gamma and k_time

Tests a grid of (gamma, k_time) combinations to find best result.
"""

import numpy as np
import json
from pathlib import Path

from core.utils.io_mat import load_binary_mask
from core.ca.model import CAParams
from core.inference.fit_bo import run_simulation

def parameter_sweep_gamma_ktime(
    location="L12",
    gamma_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    ktime_values=[0.8, 1.0, 1.2, 1.5, 2.0],
):
    """Parameter sweep for gamma and k_time."""
    print(f"\n{'='*70}")
    print(f"Parameter Sweep - Location {location}")
    print(f"  Gamma values: {gamma_values}")
    print(f"  k_time values: {ktime_values}")
    print(f"  Total combinations: {len(gamma_values) * len(ktime_values)}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)
    
    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    obs_norm = observed_wound_area / observed_wound_area[0]
    
    # Load initial mask
    print("Loading initial mask...")
    import pandas as pd
    manifest = pd.read_csv("data_manifest.csv")
    first_frame_id = ts_data["frame_ids"][0]
    row = manifest[manifest["frame_id"] == first_frame_id]
    mask_path = row["manual_mask_path"].values[0]
    
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.is_absolute():
        mask_path = str(Path(__file__).parent.parent / mask_path)
    
    initial_mask = load_binary_mask(mask_path)
    print(f"  Initial wound area: {observed_wound_area[0]:.0f}")
    print(f"  Final wound area: {observed_wound_area[-1]:.0f}\n")
    
    # Parameter sweep
    results = []
    total = len(gamma_values) * len(ktime_values)
    count = 0
    
    print("Running parameter sweep...")
    
    # Use previously optimized parameters for other parameters
    base_params = {
        'p_move': 0.8548,
        'p_div': 0.1730,
        'alpha': 1.4521,
        'beta': 0.6984,
        'edge_bonus': 1.4149,
    }
    
    for gamma in gamma_values:
        for k_time in ktime_values:
            count += 1
            print(f"[{count}/{total}] Testing gamma={gamma:.2f}, k_time={k_time:.2f}...", end=" ", flush=True)
            
            params = CAParams(
                p_move=base_params['p_move'],
                p_div=base_params['p_div'],
                alpha=base_params['alpha'],
                beta=base_params['beta'],
                edge_bonus=base_params['edge_bonus'],
                k_time=k_time,
                gamma=gamma,
            )
            
            # Run simulation
            try:
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
                
                # Interpolate to observation time points
                from scipy.interpolate import interp1d
                sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
                f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
                sim_at_obs = f_sim(observed_time)
                
                # Calculate metrics
                mse = np.mean((sim_at_obs - obs_norm)**2)
                t17_error = (sim_at_obs[1] - obs_norm[1]) * 100
                t40_error = (sim_at_obs[2] - obs_norm[2]) * 100
                
                print(f"MSE={mse:.6f}, t40_err={t40_error:+.2f}%")
                
                results.append({
                    'gamma': gamma,
                    'k_time': k_time,
                    'mse': mse,
                    't17_error': t17_error,
                    't40_error': t40_error,
                    'sim_at_obs': sim_at_obs.tolist(),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # Find best results
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}\n")
    
    # Best by MSE
    best_mse = min(results, key=lambda x: x['mse'])
    print(f"Best by MSE:")
    print(f"  gamma: {best_mse['gamma']:.3f}")
    print(f"  k_time: {best_mse['k_time']:.3f}")
    print(f"  MSE: {best_mse['mse']:.6f}")
    print(f"  t=17 error: {best_mse['t17_error']:+.2f}%")
    print(f"  t=40 error: {best_mse['t40_error']:+.2f}%")
    
    # Best by t=40 error
    best_t40 = min(results, key=lambda x: x['t40_error'])
    print(f"\nBest by t=40 error:")
    print(f"  gamma: {best_t40['gamma']:.3f}")
    print(f"  k_time: {best_t40['k_time']:.3f}")
    print(f"  MSE: {best_t40['mse']:.6f}")
    print(f"  t=17 error: {best_t40['t17_error']:+.2f}%")
    print(f"  t=40 error: {best_t40['t40_error']:+.2f}%")
    
    # Compare with baseline
    baseline_mse = 0.017
    baseline_t40_error = 61.8
    
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH BASELINE")
    print(f"{'='*70}")
    print(f"Baseline:")
    print(f"  MSE: {baseline_mse:.6f}")
    print(f"  t=40 error: +{baseline_t40_error:.2f}%")
    print(f"\nBest (by t=40 error):")
    print(f"  MSE improvement: {(baseline_mse - best_t40['mse'])/baseline_mse*100:+.1f}%")
    print(f"  t=40 improvement: {(baseline_t40_error - best_t40['t40_error'])/baseline_t40_error*100:+.1f}%")
    
    # Success criteria
    success = best_t40['t40_error'] < 40
    print(f"\n{'='*70}")
    print(f"SUCCESS CRITERIA")
    print(f"{'='*70}")
    print(f"Target: t=40 error < 40%")
    print(f"Result: {best_t40['t40_error']:+.2f}% - {'PASS' if success else 'FAIL'}")
    
    # Save results
    output_path = Path("results/parameter_sweep_gamma_ktime.json")
    with open(output_path, 'w') as f:
        json.dump({
            'location': location,
            'gamma_values': gamma_values,
            'ktime_values': ktime_values,
            'results': results,
            'best_by_mse': best_mse,
            'best_by_t40': best_t40,
            'baseline': {
                'mse': baseline_mse,
                't40_error': baseline_t40_error,
            },
            'success': success,
        }, f, indent=2)
    print(f"\nSaved results: {output_path}")
    
    return results

if __name__ == "__main__":
    results = parameter_sweep_gamma_ktime(
        location="L12",
        gamma_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        ktime_values=[0.8, 1.0, 1.2, 1.5, 2.0],
    )
    
    print("\n=== SUMMARY ===")
    best = min(results, key=lambda x: x['t40_error'])
    print(f"Best parameters (t=40 error):")
    print(f"  gamma = {best['gamma']:.3f}")
    print(f"  k_time = {best['k_time']:.3f}")
    print(f"  t=40 error = {best['t40_error']:+.2f}%")
