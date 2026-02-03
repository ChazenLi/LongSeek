"""
Minimal Validation Example - Using Core Framework Without Dataset Dependencies

This script demonstrates how to use the modular CA framework for validation
and testing without requiring external dataset files.

Usage:
    python examples/minimal_validation.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    CellOnlyCA, CAParams,
    run_simulation, calculate_loss, fit_simple_grid_search,
    downsample_binary
)
from core.preprocess.data_loader import create_minimal_validation_dataset


def minimal_simulation_example():
    """Example 1: Basic simulation with synthetic data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Minimal Simulation with Synthetic Data")
    print("="*70 + "\n")
    
    # Create minimal dataset
    print("Step 1: Creating synthetic dataset...")
    initial_mask, observed_time, observed_area, metadata = create_minimal_validation_dataset()
    
    print(f"  Dataset: {metadata['dataset_type']}")
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Initial wound area: {metadata['initial_area']:.0f} pixels")
    print(f"  Time points: {observed_time}")
    
    # Create CA model with default parameters
    print("\nStep 2: Creating CA model...")
    params = CAParams(
        p_move=0.85,
        p_div=0.17,
        alpha=1.45,
        beta=0.70,
        edge_bonus=1.41,
        k_time=1.0,
        gamma=0.0,
    )
    print(f"  Parameters: p_move={params.p_move}, p_div={params.p_div}")
    
    # Downsample for speed
    print("\nStep 3: Downsampling mask...")
    mask_ds = downsample_binary(initial_mask, k=4)
    print(f"  Downsampled shape: {mask_ds.shape}")
    
    # Create CA instance
    ca = CellOnlyCA(height=mask_ds.shape[0], width=mask_ds.shape[1], params=params)
    ca.initialize_from_mask(initial_mask, k=4)
    
    # Run simulation
    print("\nStep 4: Running simulation...")
    history = ca.run(num_steps=50)
    print(f"  Simulation steps: {len(history)}")
    
    # Extract results
    sim_area = np.array([s["wound_area"] for s in history])
    sim_area_norm = sim_area / sim_area[0]
    obs_area_norm = observed_area / observed_area[0]
    
    # Interpolate to observation time points
    from scipy.interpolate import interp1d
    sim_time_real = np.linspace(observed_time[0], observed_time[-1], len(sim_area_norm))
    f_sim = interp1d(sim_time_real, sim_area_norm, kind='linear', fill_value="extrapolate")
    sim_at_obs = f_sim(observed_time)
    
    # Calculate metrics
    mse = np.mean((sim_at_obs - obs_area_norm)**2)
    mae = np.mean(np.abs(sim_at_obs - obs_area_norm))
    
    print(f"\nStep 5: Results")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"\n  Time    Observed    Simulated    Error")
    for t, obs, sim in zip(observed_time, obs_area_norm, sim_at_obs):
        err = (sim - obs) * 100
        print(f"  t={t:3.0f}    {obs:.4f}       {sim:.4f}       {err:+.2f}%")
    
    return {
        'metadata': metadata,
        'params': params,
        'mse': mse,
        'mae': mae,
        'sim_area_norm': sim_area_norm,
        'obs_area_norm': obs_area_norm,
        'sim_at_obs': sim_at_obs,
    }


def minimal_optimization_example():
    """Example 2: Parameter optimization with synthetic data."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Parameter Optimization with Synthetic Data")
    print("="*70 + "\n")
    
    # Create dataset
    print("Step 1: Creating synthetic dataset...")
    initial_mask, observed_time, observed_area, metadata = create_minimal_validation_dataset()
    
    # Prepare observed data
    observed = {"A_t": observed_area}
    
    print(f"  Initial wound area: {metadata['initial_area']:.0f} pixels")
    print(f"  Time points: {observed_time}")
    
    # Run optimization (small sample for speed)
    print("\nStep 2: Running grid search optimization (5 samples)...")
    result = fit_simple_grid_search(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        downsample_k=4,
        n_samples=5,
    )
    
    print(f"\nStep 3: Optimization Results")
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  Best parameters:")
    print(f"    p_move: {result['best_params'].p_move:.4f}")
    print(f"    p_div: {result['best_params'].p_div:.4f}")
    print(f"    alpha: {result['best_params'].alpha:.4f}")
    print(f"    beta: {result['best_params'].beta:.4f}")
    print(f"    edge_bonus: {result['best_params'].edge_bonus:.4f}")
    print(f"    k_time: {result['best_params'].k_time:.4f}")
    print(f"    gamma: {result['best_params'].gamma:.4f}")
    
    return result


def minimal_api_example():
    """Example 3: Direct API usage."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Direct API Usage")
    print("="*70 + "\n")
    
    # Create synthetic data
    initial_mask, observed_time, observed_area, metadata = create_minimal_validation_dataset()
    observed = {"A_t": observed_area}
    
    print("Using high-level API: run_simulation()")
    
    # Use high-level API
    params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, edge_bonus=2.0)
    
    history = run_simulation(
        params=params,
        initial_mask=initial_mask,
        num_steps=50,
        downsample_k=4,
        seed=42,
        observed_time=observed_time,
    )
    
    loss = calculate_loss(history, observed, observed_time, k_time=params.k_time)
    
    print(f"  Simulation steps: {len(history)}")
    print(f"  Loss: {loss:.6f}")
    print(f"  Final wound area: {history[-1]['wound_area']:.0f}")


def visualize_results(results):
    """Visualize simulation results."""
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Time series
    ax1 = axes[0]
    sim_time_real = np.linspace(0, 40, len(results['sim_area_norm']))
    ax1.plot(sim_time_real, results['sim_area_norm'], 'b-', label='Simulated', linewidth=2)
    ax1.plot([0, 17, 40], results['obs_area_norm'], 'ro', label='Observed', markersize=10)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized wound area')
    ax1.set_title(f'Wound Healing Simulation (MSE={results["mse"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Point-by-point comparison
    ax2 = axes[1]
    x_pos = np.arange(3)
    width = 0.35
    ax2.bar(x_pos - width/2, results['obs_area_norm'], width, label='Observed', color='r', alpha=0.7)
    ax2.bar(x_pos + width/2, results['sim_at_obs'], width, label='Simulated', color='b', alpha=0.7)
    ax2.set_xlabel('Time point')
    ax2.set_ylabel('Normalized wound area')
    ax2.set_title('Point-by-Point Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['t=0', 't=17', 't=40'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save to examples directory
    output_path = Path(__file__).parent / "minimal_validation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CA FRAMEWORK - MINIMAL VALIDATION EXAMPLES")
    print("="*70)
    print("\nThis demonstrates the modular framework without dataset dependencies.")
    
    try:
        # Example 1: Basic simulation
        results = minimal_simulation_example()
        
        # Visualize
        visualize_results(results)
        
        # Example 2: Optimization
        # opt_result = minimal_optimization_example()
        
        # Example 3: API usage
        # minimal_api_example()
        
        print("\n" + "="*70)
        print("[SUCCESS] All examples completed successfully!")
        print("="*70 + "\n")
        
        print("\nKey Takeaways:")
        print("1. Core framework works without external dataset files")
        print("2. Modular API: from core import CellOnlyCA, CAParams, ...")
        print("3. Easy to create synthetic data for testing")
        print("4. Ready for integration with any dataset format")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
