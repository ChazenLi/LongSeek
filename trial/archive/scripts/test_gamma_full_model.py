"""
Complete CA Model Test with Gamma Parameter
Tests the directional bias hypothesis on full CA model (not simplified)
"""

import numpy as np
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from io_mat import load_binary_mask
from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import run_simulation, calculate_loss
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def test_gamma_full_ca_model(location="L12", gamma_values=[0.0, 0.5, 1.0]):
    """
    Test gamma parameter on full CA model
    """
    print(f"\n{'='*70}")
    print(f"Full CA Model Test - Directional Bias (gamma)")
    print(f"Location: {location}")
    print(f"{'='*70}\n")
    
    # Load data
    with open(f"time_series_{location}.json", 'r') as f:
        ts_data = json.load(f)
    
    observed_time = np.array(ts_data["time_points"])
    observed_wound_area = np.array(ts_data["wound_area"])
    observed = {"A_t": observed_wound_area}
    
    print(f"Observation data:")
    print(f"  Time points: {observed_time}")
    print(f"  Wound areas: {observed_wound_area}")
    print(f"  Closure rate: {(1 - observed_wound_area[-1]/observed_wound_area[0])*100:.1f}%")
    
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
    print(f"\nInitial mask loaded: {initial_mask.shape}")
    
    # Test base parameters (optimized from before)
    base_params = {
        'p_move': 0.85,
        'p_div': 0.17,
        'alpha': 1.45,
        'beta': 0.70,
        'edge_bonus': 1.41,
    }
    
    print(f"\nTesting gamma values: {gamma_values}")
    print(f"Base parameters: p_move={base_params['p_move']}, p_div={base_params['p_div']}")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, gamma in enumerate(gamma_values):
        print(f"\n[{i+1}/{len(gamma_values)}] Testing gamma={gamma}")
        
        # Create parameters
        params = CAParams(
            p_move=base_params['p_move'],
            p_div=base_params['p_div'],
            alpha=base_params['alpha'],
            beta=base_params['beta'],
            edge_bonus=base_params['edge_bonus'],
            k_time=1.0,
            gamma=gamma
        )
        
        # Run simulation
        start_time = time.time()
        
        try:
            history = run_simulation(
                params=params,
                initial_mask=initial_mask,
                num_steps=50,
                downsample_k=4,
                seed=42,
                observed_time=observed_time,
            )
            
            elapsed = time.time() - start_time
            
            # Extract results
            simulated_wound_area = np.array([s["wound_area"] for s in history])
            sim_norm = simulated_wound_area / simulated_wound_area[0]
            obs_norm = observed_wound_area / observed_wound_area[0]
            
            # Interpolate to observation time points
            sim_time_real = np.linspace(0, observed_time[-1], len(sim_norm))
            f_sim = interp1d(sim_time_real, sim_norm, kind='linear', bounds_error=False, fill_value="extrapolate")
            sim_at_obs = f_sim(observed_time)
            
            # Calculate metrics
            mse = np.mean((sim_at_obs - obs_norm)**2)
            mae = np.mean(np.abs(sim_at_obs - obs_norm))
            
            # Point-wise errors
            errors = (sim_at_obs - obs_norm) * 100
            
            print(f"  Simulation time: {elapsed:.1f}s")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"\n  Point-wise comparison:")
            for j, (t, obs, sim, err) in enumerate(zip(observed_time, obs_norm, sim_at_obs, errors)):
                print(f"    t={t:3.0f}h: obs={obs:.4f}, sim={sim:.4f}, error={err:+.2f}%")
            
            results.append({
                'gamma': gamma,
                'mse': mse,
                'mae': mae,
                'sim_at_obs': sim_at_obs.tolist(),
                'errors': errors.tolist(),
                'elapsed': elapsed
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Find best result
    if results:
        best = min(results, key=lambda x: x['mse'])
        worst = max(results, key=lambda x: x['mse'])
        
        print(f"\n{'='*70}")
        print(f"BEST RESULT:")
        print(f"  Gamma: {best['gamma']}")
        print(f"  MSE: {best['mse']:.6f}")
        print(f"  MAE: {best['mae']:.6f}")
        print(f"\n  Errors:")
        for j, (t, err) in enumerate(zip(observed_time, best['errors'])):
            print(f"    t={t:3.0f}h: {err:+.2f}%")
        
        print(f"\n  Improvement over gamma=0:")
        baseline_err = results[0]['errors'][2]  # t=40 error with gamma=0
        best_err = best['errors'][2]
        improvement = baseline_err - best_err
        print(f"    t=40 error: {baseline_err:+.1f}% → {best_err:+.1f}%")
        print(f"    Improvement: {improvement:.1f} percentage points")
        
        print(f"\nCOMPARISON:")
        print(f"  Current (gamma=0): MSE={results[0]['mse']:.6f}, t=40 err={results[0]['errors'][2]:+.1f}%")
        print(f"  Best (gamma={best['gamma']}): MSE={best['mse']:.6f}, t=40 err={best['errors'][2]:+.1f}%")
        
        if best['gamma'] > 0:
            improvement_mse = (results[0]['mse'] - best['mse']) / results[0]['mse'] * 100
            print(f"\n  ✓ HYPOTHESIS CONFIRMED:")
            print(f"    Directional bias (gamma={best['gamma']}) improves fit!")
            print(f"    MSE reduced by {improvement_mse:.1f}%")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT SUPPORTED:")
            print(f"    gamma=0 is still best")
        
        print(f"{'='*70}\n")
        
        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        results_path = output_dir / "gamma_full_test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'location': location,
                'gamma_values_tested': gamma_values,
                'results': results,
                'best_gamma': best['gamma'],
                'best_mse': best['mse'],
                'baseline_mse': results[0]['mse']
            }, f, indent=2)
        
        print(f"Results saved: {results_path}")
        
        # Create visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Comparison of wound area curves
            ax1 = axes[0, 0]
            for result in results:
                ax1.plot(observed_time, result['sim_at_obs'], 'o-', 
                       label=f'gamma={result["gamma"]}', linewidth=2, markersize=8)
            ax1.plot(observed_time, obs_norm, 'r--', label='Observed', linewidth=3, marker='s', markersize=10)
            ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Normalized Wound Area', fontsize=12, fontweight='bold')
            ax1.set_title('Effect of Directional Bias on Full CA Model', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: MSE vs gamma
            ax2 = axes[0, 1]
            gammas_tested = [r['gamma'] for r in results]
            mses = [r['mse'] for r in results]
            ax2.plot(gammas_tested, mses, 'bo-', linewidth=2, markersize=10)
            ax2.set_xlabel('Gamma (directional bias)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
            ax2.set_title('MSE vs Directional Bias Strength', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, max(mses) * 1.1])
            
            # Plot 3: t=40 error vs gamma
            ax3 = axes[1, 0]
            t40_errors = [r['errors'][2] for r in results]
            ax3.plot(gammas_tested, t40_errors, 'ro-', linewidth=2, markersize=10)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
            ax3.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target zone')
            ax3.set_xlabel('Gamma (directional bias)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Error at t=40 (%)', fontsize=12, fontweight='bold')
            ax3.set_title('Late-Stage Error vs Directional Bias', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.fill_between(gammas_tested, -20, 20, color='lightgreen', alpha=0.3)
            
            # Plot 4: Bar chart of errors at all time points
            ax4 = axes[1, 1]
            x = np.arange(len(observed_time))
            width = 0.15
            
            for i, gamma in enumerate(gammas_tested):
                errors = results[i]['errors']
                ax4.bar(x + i*width, errors, width, label=f'γ={gamma}', alpha=0.7)
            
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax4.set_xlabel('Time point', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
            ax4.set_title('Error Distribution Across Time Points', fontsize=14, fontweight='bold')
            ax4.set_xticks(x + width * (len(gammas_tested) - 1) / 2)
            ax4.set_xticklabels([f't={t}' for t in observed_time], fontsize=10)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim([-30, 80])
            
            plt.tight_layout()
            
            plot_path = output_dir / "gamma_full_ca_test.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return results
    
    else:
        print("No results obtained!")
        return None

if __name__ == "__main__":
    # Test with smaller set first
    results = test_gamma_full_ca_model(
        location="L12",
        gamma_values=[0.0, 0.5, 1.0]
    )
    
    if results:
        print("\n=== KEY FINDINGS ===")
        print("1. Full CA model test completed")
        print("2. Results saved to results/gamma_full_test_results.json")
        print("3. Visualization saved to results/gamma_full_ca_test.png")
        print("\n=== NEXT STEP ===")
        print("Update optimization scripts to include gamma in search space")
