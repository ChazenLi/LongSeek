"""
Simplified end-to-end example using synthetic data.
This demonstrates the CA modeling pipeline without requiring original data files.
"""
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import run_simulation, objective_function, fit_simple_grid_search
import matplotlib.pyplot as plt

def create_synthetic_mask(size=256, wound_radius=80):
    """Create a synthetic wound healing mask."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    # Circular wound
    wound = (x - center)**2 + (y - center)**2 < wound_radius**2
    # Cells around wound
    cells = ~wound
    # Add some noise
    noise = np.random.random((size, size)) > 0.95
    cells = cells | noise
    return cells.astype(np.int32)

def create_synthetic_observed_data(num_steps=40):
    """Create synthetic observed wound area time series."""
    # Simulated wound healing: exponential decay
    time_points = np.linspace(0, 40, num_steps)
    initial_area = 50000
    # Wound area decreases over time
    wound_area = initial_area * np.exp(-0.03 * time_points)
    # Add some noise
    wound_area += np.random.normal(0, 500, size=num_steps)
    wound_area = np.maximum(wound_area, 5000)  # Minimum wound area

    return {
        "time_points": time_points,
        "wound_area": wound_area,
        "A_t": wound_area,  # For compatibility
    }

def plot_comparison(
    simulated_wound_area: np.ndarray,
    observed_wound_area: np.ndarray,
    title: str = "Wound Area: Simulated vs Observed",
    save_path: str = None
):
    """Plot comparison between simulated and observed data."""
    plt.figure(figsize=(10, 6))

    # Normalize to initial value
    sim_norm = simulated_wound_area / simulated_wound_area[0]
    obs_norm = observed_wound_area / observed_wound_area[0]

    plt.plot(obs_norm, 'r--', label='Observed (Synthetic)', linewidth=2, marker='o', markersize=4)
    plt.plot(sim_norm, 'b-', label='Simulated', linewidth=2)

    plt.xlabel('Time step')
    plt.ylabel('Normalized wound area')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def run_synthetic_example(
    grid_size: int = 128,
    num_steps: int = 40,
    n_samples: int = 10
):
    """
    Run end-to-end example with synthetic data.

    Args:
        grid_size: Size of CA grid
        num_steps: Number of simulation steps
        n_samples: Number of parameter samples for fitting
    """
    print(f"\n{'='*60}")
    print(f"Synthetic CA Modeling Example")
    print(f"{'='*60}\n")

    # Step 1: Create synthetic data
    print("Step 1: Creating synthetic data...")
    initial_mask = create_synthetic_mask(size=256, wound_radius=80)
    print(f"  Initial mask shape: {initial_mask.shape}")
    print(f"  Cell coverage: {np.mean(initial_mask):.3f}")
    print(f"  Wound area: {np.sum(initial_mask == 0):.0f}")

    # Step 2: Downsample mask
    print("\nStep 2: Downsampling mask...")
    from preprocess.extract_observations import downsample_binary
    downsample_k = 2
    mask_ds = downsample_binary(initial_mask, k=downsample_k, thr=0.5)
    print(f"  Original shape: {initial_mask.shape}")
    print(f"  Downsampled shape: {mask_ds.shape}")
    print(f"  Downsampling factor: {downsample_k}x{downsample_k}")

    # Step 3: Create synthetic observed data
    print("\nStep 3: Creating synthetic observed data...")
    observed = create_synthetic_observed_data(num_steps)
    observed_time = observed["time_points"]
    observed_wound_area = observed["wound_area"]
    print(f"  Time points: {len(observed_time)}")
    print(f"  Initial wound area: {observed_wound_area[0]:.0f}")
    print(f"  Final wound area: {observed_wound_area[-1]:.0f}")

    # Step 4: Fit CA parameters
    print(f"\nStep 4: Fitting CA parameters ({n_samples} samples)...")
    result = fit_simple_grid_search(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        downsample_k=downsample_k,
        n_samples=n_samples,
    )

    print(f"\n  Best loss: {result['best_loss']:.4f}")
    print(f"  Best parameters:")
    print(f"    p_move: {result['best_params'].p_move:.4f}")
    print(f"    p_div: {result['best_params'].p_div:.4f}")
    print(f"    alpha: {result['best_params'].alpha:.4f}")
    print(f"    beta: {result['best_params'].beta:.4f}")
    print(f"    edge_bonus: {result['best_params'].edge_bonus:.4f}")

    # Step 5: Run final simulation with best parameters
    print("\nStep 5: Running final simulation with best parameters...")
    history = run_simulation(
        params=result['best_params'],
        initial_mask=initial_mask,
        num_steps=num_steps,
        downsample_k=downsample_k,
        seed=42,
    )

    simulated_wound_area = np.array([s["wound_area"] for s in history])
    print(f"  Simulated wound area: {simulated_wound_area}")

    # Step 6: Compare results
    print("\nStep 6: Comparing results...")
    print(f"  Observed final wound area: {observed_wound_area[-1]:.0f}")
    print(f"  Simulated final wound area: {simulated_wound_area[-1]:.0f}")
    print(f"  Observed closure rate: {(observed_wound_area[0] - observed_wound_area[-1]) / len(observed_wound_area):.2f}")
    print(f"  Simulated closure rate: {(simulated_wound_area[0] - simulated_wound_area[-1]) / len(simulated_wound_area):.2f}")

    # Step 7: Plot results
    print("\nStep 7: Plotting results...")
    output_dir = Path("CA_project/results")
    output_dir.mkdir(exist_ok=True)

    plot_path = output_dir / "comparison_synthetic.png"
    plot_comparison(
        simulated_wound_area=simulated_wound_area,
        observed_wound_area=observed_wound_area,
        title="Wound Area: Simulated vs Observed (Synthetic Data)",
        save_path=str(plot_path),
    )

    # Step 8: Save results
    print("\nStep 8: Saving results...")
    results_path = output_dir / "results_synthetic.json"
    results = {
        "best_params": {
            "p_move": float(result['best_params'].p_move),
            "p_div": float(result['best_params'].p_div),
            "alpha": float(result['best_params'].alpha),
            "beta": float(result['best_params'].beta),
            "edge_bonus": float(result['best_params'].edge_bonus),
        },
        "best_loss": float(result['best_loss']),
        "observed_wound_area": observed_wound_area.tolist(),
        "simulated_wound_area": simulated_wound_area.tolist(),
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved results to {results_path}")

    print(f"\n{'='*60}")
    print("Synthetic example completed successfully!")
    print(f"{'='*60}\n")

    return results

if __name__ == "__main__":
    # Run synthetic example
    results = run_synthetic_example(
        grid_size=128,
        num_steps=40,
        n_samples=10,  # Small number for quick demo
    )

    print("\n=== Summary ===")
    print(f"Best parameters:")
    for k, v in results['best_params'].items():
        print(f"  {k}: {v:.4f}")
    print(f"Best loss: {results['best_loss']:.4f}")
