import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from io_mat import load_binary_mask
from preprocess.extract_observations import extract_frame_statistics, downsample_binary
from ca.model import CellOnlyCA, CAParams
from inference.fit_bo import run_simulation, objective_function, fit_simple_grid_search
import matplotlib.pyplot as plt

def load_time_series(location: str) -> dict:
    """Load time series for a specific location."""
    with open(f"time_series_{location}.json", 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    data["time_points"] = np.array(data["time_points"])
    data["wound_area"] = np.array(data["wound_area"])
    data["cell_coverage"] = np.array(data["cell_coverage"])
    
    return data

def get_initial_mask(location: str, mask_type: str = "manual_mask_path") -> np.ndarray:
    """
    Load the first frame mask for a location.
    
    Args:
        location: Location identifier (e.g., "L12")
        mask_type: Which mask type to use
        
    Returns:
        Binary mask (1=cell, 0=empty)
    """
    # Load comparison data to find frame path
    with open("comparison_dataset.json", 'r') as f:
        comparison = json.load(f)
    
    # Find the frame for this location
    ts_data = load_time_series(location)
    first_frame_id = ts_data["frame_ids"][0]
    
    # Load the mask
    if mask_type == "manual_mask_path":
        # Extract the actual path
        manifest_path = "data_manifest.csv"
        import pandas as pd
        manifest = pd.read_csv(manifest_path)
        
        row = manifest[manifest["frame_id"] == first_frame_id]
        if len(row) > 0:
            mask_path = row[mask_type].values[0]
            mask = load_binary_mask(mask_path)
            return mask
    
    raise ValueError(f"Could not load mask for {location} using {mask_type}")

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
    
    plt.plot(sim_norm, 'b-', label='Simulated', linewidth=2)
    plt.plot(obs_norm, 'r--', label='Observed', linewidth=2, marker='o')
    
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

def run_end_to_end_example(
    location: str = "L12",
    downsample_k: int = 4,
    num_steps: int = 100,
    n_samples: int = 20
):
    """
    Run end-to-end example: load data, fit CA, plot results.
    
    Args:
        location: Location identifier
        downsample_k: Downsampling factor for CA grid
        num_steps: Number of simulation steps
        n_samples: Number of random samples for parameter fitting
    """
    print(f"\n{'='*60}")
    print(f"End-to-End CA Modeling Example: Location {location}")
    print(f"{'='*60}\n")
    
    # Step 1: Load time series data
    print("Step 1: Loading time series data...")
    ts_data = load_time_series(location)
    print(f"  Location: {ts_data['location']}")
    print(f"  Frames: {len(ts_data['frame_ids'])}")
    print(f"  Time range: {ts_data['time_points'][0]} to {ts_data['time_points'][-1]}")
    print(f"  Initial wound area: {ts_data['wound_area'][0]:.0f}")
    print(f"  Final wound area: {ts_data['wound_area'][-1]:.0f}")
    
    # Step 2: Load initial mask
    print("\nStep 2: Loading initial mask...")
    initial_mask = get_initial_mask(location, mask_type="manual_mask_path")
    print(f"  Mask shape: {initial_mask.shape}")
    print(f"  Cell coverage: {np.mean(initial_mask):.3f}")
    print(f"  Wound area: {np.sum(initial_mask == 0):.0f}")
    
    # Step 3: Downsample mask
    print("\nStep 3: Downsampling mask...")
    H, W = initial_mask.shape
    H_ds, W_ds = H // downsample_k, W // downsample_k
    mask_ds = downsample_binary(initial_mask, k=downsample_k, thr=0.5)
    print(f"  Original shape: {initial_mask.shape}")
    print(f"  Downsampled shape: {mask_ds.shape}")
    print(f"  Downsampling factor: {downsample_k}x{downsample_k}")
    
    # Step 4: Prepare observed data for fitting
    print("\nStep 4: Preparing observed data...")
    # Interpolate observed data to match simulation steps
    observed_wound_area = ts_data["wound_area"]
    observed_time = ts_data["time_points"]
    
    # Calculate total time and create time points
    total_time = observed_time[-1] - observed_time[0]
    print(f"  Total time interval: {total_time}")
    print(f"  Time points: {observed_time}")
    
    # Run simulation for total_time steps (each step = 1 time unit)
    # Cap at 50 steps to avoid long runtime
    num_steps = min(int(total_time) + 1, 50)
    print(f"  Number of simulation steps: {num_steps}")
    
    # Create observed dictionary (use raw wound areas, will normalize in loss)
    observed = {
        "A_t": observed_wound_area,
    }
    print(f"  Observed wound area: {observed_wound_area}")
    
    # Step 5: Fit CA parameters
    print("\nStep 5: Fitting CA parameters...")
    result = fit_simple_grid_search(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        num_steps=num_steps,
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
    
    # Step 6: Run final simulation with best parameters
    print("\nStep 6: Running final simulation with best parameters...")
    history = run_simulation(
        params=result['best_params'],
        initial_mask=initial_mask,
        num_steps=num_steps,
        downsample_k=downsample_k,
        seed=42,
    )
    
    simulated_wound_area = np.array([s["wound_area"] for s in history])
    print(f"  Simulated wound area: {simulated_wound_area}")
    
    # Step 7: Compare results
    print("\nStep 7: Comparing results...")
    print(f"  Observed final wound area: {observed_wound_area[-1]:.0f}")
    print(f"  Simulated final wound area: {simulated_wound_area[-1]:.0f}")
    print(f"  Observed closure rate: {(observed_wound_area[0] - observed_wound_area[-1]) / len(observed_wound_area):.2f}")
    print(f"  Simulated closure rate: {(simulated_wound_area[0] - simulated_wound_area[-1]) / len(simulated_wound_area):.2f}")
    
    # Step 8: Plot results
    print("\nStep 8: Plotting results...")
    output_dir = Path("CA_project/results")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / f"comparison_{location}.png"
    plot_comparison(
        simulated_wound_area=simulated_wound_area,
        observed_wound_area=observed_wound_area,
        title=f"Wound Area: Simulated vs Observed ({location})",
        save_path=str(plot_path),
    )
    
    # Step 9: Save results
    print("\nStep 9: Saving results...")
    results_path = output_dir / f"results_{location}.json"
    results = {
        "location": location,
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
    print("Example completed successfully!")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    # Run example for location L12
    results = run_end_to_end_example(
        location="L12",
        downsample_k=4,
        n_samples=5,  # Reduce for faster testing
    )
    
    print("\n=== Summary ===")
    print(f"Location: {results['location']}")
    print(f"Best parameters:")
    for k, v in results['best_params'].items():
        print(f"  {k}: {v:.4f}")
    print(f"Best loss: {results['best_loss']:.4f}")
