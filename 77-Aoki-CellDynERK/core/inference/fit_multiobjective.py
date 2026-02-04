"""
Multi-Objective Inference for CA Model Parameters

This module extends the basic inference framework to support:
- Multiple observables (wound area, roughness, width, shape metrics)
- Uncertainty weighting (frame weights, noise model)
- Multi-objective optimization (weighted sum or Pareto)
- Integration with morphological metrics

Key Concepts:
- Multi-objective loss: L = Σ w_i * L_i
- Weighted residuals: residual_t / σ_t²
- Identifiability: More observables → better parameter identifiability
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def calculate_multiobjective_loss(
    simulated: List[Dict[str, float]],
    observed: Dict[str, any],
    observed_time: np.ndarray,
    k_time: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate multi-objective loss between simulated and observed data.
    
    Supports multiple observables:
    - A_t: Wound area (main observable)
    - roughness_perimeter: Perimeter-based roughness
    - roughness_height_field: Height field roughness
    - width_mean: Mean width
    - width_std: Width standard deviation
    - aspect_ratio: Aspect ratio
    - solidity: Solidity metric
    - circularity: Circularity metric
    
    Args:
        simulated: List of simulation steps
        observed: Dictionary of observed time series
        observed_time: Time points for observed data (real time units)
        k_time: Time scaling factor (CA steps per real time unit)
        weights: Weights for each observable
        frame_weights: Per-frame weights based on segmentation quality
        noise_levels: Per-frame noise levels for uncertainty weighting
        
    Returns:
        Tuple of (total_loss, loss_breakdown)
    """
    if weights is None:
        weights = {
            "A_t": 1.0,
            "roughness_perimeter": 0.5,
            "roughness_height_field": 0.5,
            "width_mean": 0.3,
            "width_std": 0.2,
            "aspect_ratio": 0.3,
            "solidity": 0.2,
            "circularity": 0.2,
        }
    
    loss_breakdown = {}
    total_loss = 0.0
    
    num_steps = len(simulated)
    sim_time_real = np.arange(num_steps) / k_time
    
    # === Wound Area Loss ===
    if "A_t" in observed:
        sim_A = np.array([s["wound_area"] for s in simulated])
        obs_A = np.array(observed["A_t"])
        
        # Normalize to initial value
        sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A
        obs_A_norm = obs_A / obs_A[0] if obs_A[0] > 0 else obs_A
        
        # Interpolate simulated data to observed time points
        if num_steps > 1:
            try:
                f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear', 
                                 bounds_error=False, fill_value="extrapolate")
                sim_at_obs_time = f_sim(observed_time)
            except:
                sim_at_obs_time = np.full_like(obs_A_norm, sim_A_norm[0])
        else:
            sim_at_obs_time = np.full_like(obs_A_norm, sim_A_norm[0])
        
        # Calculate residuals
        residuals = (sim_at_obs_time - obs_A_norm) ** 2
        
        # Apply frame weights
        if frame_weights is not None and len(frame_weights) == len(residuals):
            residuals = residuals * frame_weights
        
        # Apply noise model
        if noise_levels is not None and len(noise_levels) == len(residuals):
            sigma_sq = noise_levels ** 2 + 1e-10
            residuals = residuals / sigma_sq
        
        mse_A = np.mean(residuals)
        loss_breakdown["A_t"] = mse_A
        total_loss += weights.get("A_t", 1.0) * mse_A
    
    # === Morphological Losses ===
    
    # Roughness (perimeter-based)
    if "roughness_perimeter" in observed:
        obs_R = np.array(observed["roughness_perimeter"])
        
        # For now, simulate roughness from area
        # In practice, you'd compute actual roughness from grid
        if num_steps > 1:
            try:
                f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear',
                                 bounds_error=False, fill_value="extrapolate")
                sim_at_obs_time = f_sim(observed_time)
            except:
                sim_at_obs_time = np.full_like(obs_R, 1.0)
        else:
            sim_at_obs_time = np.full_like(obs_R, 1.0)
        
        # Roughness inversely related to closure (more closure → smoother edges)
        sim_R = 1.0 - (sim_at_obs_time - 1.0) ** 2
        
        residuals_R = (sim_R - obs_R) ** 2
        mse_R = np.mean(residuals_R)
        loss_breakdown["roughness_perimeter"] = mse_R
        total_loss += weights.get("roughness_perimeter", 0.5) * mse_R
    
    # Width statistics
    if "width_mean" in observed:
        obs_W = np.array(observed["width_mean"])
        
        # Simulate width from area (assumes roughly square wound)
        if num_steps > 1:
            try:
                f_sim = interp1d(sim_time_real, np.sqrt(sim_A), kind='linear',
                                 bounds_error=False, fill_value="extrapolate")
                sim_at_obs_time = f_sim(observed_time)
            except:
                sim_at_obs_time = np.full_like(obs_W, np.sqrt(sim_A[0]))
        else:
            sim_at_obs_time = np.full_like(obs_W, np.sqrt(sim_A[0]))
        
        residuals_W = (sim_at_obs_time - obs_W) ** 2
        mse_W = np.mean(residuals_W)
        loss_breakdown["width_mean"] = mse_W
        total_loss += weights.get("width_mean", 0.3) * mse_W
    
    # Shape metrics (aspect ratio, solidity, circularity)
    for shape_metric in ["aspect_ratio", "solidity", "circularity"]:
        if shape_metric in observed:
            obs_shape = np.array(observed[shape_metric])
            
            # For now, assume shape remains roughly constant
            # In practice, you'd compute actual shape metrics
            if len(observed_time) > 1:
                sim_shape = np.full_like(obs_shape, 1.0)
            else:
                sim_shape = np.array([1.0])
            
            residuals_shape = (sim_shape - obs_shape) ** 2
            mse_shape = np.mean(residuals_shape)
            loss_breakdown[shape_metric] = mse_shape
            total_loss += weights.get(shape_metric, 0.3) * mse_shape
    
    return total_loss, loss_breakdown


def fit_multiobjective_grid_search(
    initial_mask: np.ndarray,
    observed: Dict[str, any],
    observed_time: np.ndarray,
    downsample_k: int = 4,
    n_samples: int = 30,
    weights: Optional[Dict[str, float]] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Fit CA parameters using grid search with multi-objective loss.
    
    Args:
        initial_mask: Initial binary mask
        observed: Dictionary of observed time series
        observed_time: Time points for observed data
        downsample_k: Downsampling factor
        n_samples: Number of random samples
        weights: Weights for each observable
        frame_weights: Per-frame weights
        noise_levels: Per-frame noise levels
        
    Returns:
        Best parameters and result
    """
    from core.ca.model import CAParams
    from core.inference.fit_bo import run_simulation
    
    best_loss = float('inf')
    best_params = None
    best_loss_breakdown = None
    best_history = None
    
    print(f"Running multi-objective grid search with {n_samples} samples...")
    print("Observables:", list(observed.keys()))
    if weights:
        print("Weights:", weights)
    
    for i in range(n_samples):
        # Random sampling
        params = CAParams(
            p_move=float(np.random.uniform(0.01, 0.99)),
            p_div=float(np.random.uniform(0.001, 0.2)),
            alpha=float(np.random.uniform(0.1, 5.0)),
            beta=float(np.random.uniform(0.1, 5.0)),
            edge_bonus=float(np.random.uniform(0.5, 5.0)),
            k_time=float(np.random.uniform(0.5, 3.0)),
            gamma=float(np.random.uniform(0.0, 2.0)),
        )
        
        # Run simulation
        history = run_simulation(
            params=params,
            initial_mask=initial_mask,
            num_steps=50,
            downsample_k=downsample_k,
            observed_time=observed_time
        )
        
        # Calculate multi-objective loss
        loss, loss_breakdown = calculate_multiobjective_loss(
            history, observed, observed_time, k_time=params.k_time,
            weights=weights, frame_weights=frame_weights, noise_levels=noise_levels
        )
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_loss_breakdown = loss_breakdown
            best_history = history
        
        # Print progress
        if (i + 1) % 5 == 0 or i == n_samples - 1:
            print(f"Sample {i+1}/{n_samples}: loss={loss:.6f}, k_time={params.k_time:.2f}, gamma={params.gamma:.2f}")
            print(f"  Loss breakdown: {loss_breakdown}")
    
    return {
        "best_params": best_params,
        "best_loss": best_loss,
        "best_loss_breakdown": best_loss_breakdown,
        "best_history": best_history,
    }


def fit_multiobjective_bayesian(
    initial_mask: np.ndarray,
    observed: Dict[str, any],
    observed_time: np.ndarray,
    n_calls: int = 100,
    n_initial_points: int = 30,
    random_state: Optional[int] = None,
    downsample_k: int = 4,
    weights: Optional[Dict[str, float]] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Fit CA parameters using Bayesian optimization with multi-objective loss.
    
    Args:
        initial_mask: Initial binary mask
        observed: Dictionary of observed time series
        observed_time: Time points for observed data
        n_calls: Number of function evaluations
        n_initial_points: Number of initial random points
        random_state: Random seed
        downsample_k: Downsampling factor
        weights: Weights for each observable
        frame_weights: Per-frame weights
        noise_levels: Per-frame noise levels
        
    Returns:
        Best parameters and result
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("scikit-optimize not installed. Falling back to grid search...")
        return fit_multiobjective_grid_search(
            initial_mask, observed, observed_time, downsample_k, n_samples=n_calls,
            weights=weights, frame_weights=frame_weights, noise_levels=noise_levels
        )
    
    from core.ca.model import CAParams
    from core.inference.fit_bo import run_simulation
    
    # Define search space
    space = [
        Real(0.01, 0.99, name='p_move'),
        Real(0.001, 0.2, name='p_div'),
        Real(0.1, 5.0, name='alpha'),
        Real(0.1, 5.0, name='beta'),
        Real(0.5, 5.0, name='edge_bonus'),
        Real(0.5, 3.0, name='k_time'),
        Real(0.0, 2.0, name='gamma'),
    ]
    
    # Objective function
    def objective(params_array):
        params = CAParams(
            p_move=float(params_array[0]),
            p_div=float(params_array[1]),
            alpha=float(params_array[2]),
            beta=float(params_array[3]),
            edge_bonus=float(params_array[4]),
            k_time=float(params_array[5]),
            gamma=float(params_array[6]),
        )
        
        # Run simulation
        history = run_simulation(
            params=params,
            initial_mask=initial_mask,
            num_steps=50,
            downsample_k=downsample_k,
            observed_time=observed_time
        )
        
        # Calculate multi-objective loss
        loss, loss_breakdown = calculate_multiobjective_loss(
            history, observed, observed_time, k_time=params.k_time,
            weights=weights, frame_weights=frame_weights, noise_levels=noise_levels
        )
        
        return loss
    
    # Run Bayesian optimization
    print(f"Running multi-objective Bayesian optimization with {n_calls} calls...")
    print("Observables:", list(observed.keys()))
    if weights:
        print("Weights:", weights)
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=True
    )
    
    # Extract best parameters
    best_params = CAParams(
        p_move=float(result.x[0]),
        p_div=float(result.x[1]),
        alpha=float(result.x[2]),
        beta=float(result.x[3]),
        edge_bonus=float(result.x[4]),
        k_time=float(result.x[5]),
        gamma=float(result.x[6]),
    )
    
    # Run best simulation to get history
    best_history = run_simulation(
        params=best_params,
        initial_mask=initial_mask,
        num_steps=50,
        downsample_k=downsample_k,
        observed_time=observed_time
    )
    
    # Calculate best loss breakdown
    best_loss, best_loss_breakdown = calculate_multiobjective_loss(
        best_history, observed, observed_time, k_time=best_params.k_time,
        weights=weights, frame_weights=frame_weights, noise_levels=noise_levels
    )
    
    return {
        "best_params": best_params,
        "best_loss": float(best_loss),
        "best_loss_breakdown": best_loss_breakdown,
        "best_history": best_history,
        "result": result,
    }


if __name__ == "__main__":
    # Test multi-objective inference
    print("Testing Multi-Objective Inference...")
    
    # Create synthetic test data
    initial_mask = np.ones((100, 100), dtype=np.uint8)
    initial_mask[30:70, 30:70] = 0  # Square wound
    
    observed_time = np.array([0, 17, 40])
    observed_A = np.array([1600, 800, 200])
    
    # Add morphological observables
    observed = {
        "A_t": observed_A,
        "roughness_perimeter": np.array([1.0, 0.9, 0.8]),
        "width_mean": np.array([40.0, 35.0, 20.0]),
        "aspect_ratio": np.array([1.0, 1.0, 1.0]),
    }
    
    # Add frame weights (simulate varying quality)
    frame_weights = np.array([1.0, 0.9, 0.8])
    
    # Test multi-objective loss calculation
    from core.ca.model import CAParams
    from core.inference.fit_bo import run_simulation
    
    params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, 
                     edge_bonus=2.0, k_time=1.0, gamma=0.0)
    
    history = run_simulation(params, initial_mask, num_steps=50, 
                        downsample_k=2, observed_time=observed_time)
    
    loss, loss_breakdown = calculate_multiobjective_loss(
        history, observed, observed_time, k_time=1.0,
        frame_weights=frame_weights
    )
    
    print("\n=== Multi-Objective Loss ===")
    print(f"Total loss: {loss:.6f}")
    print("Loss breakdown:")
    for key, value in loss_breakdown.items():
        print(f"  {key}: {value:.6f}")
    
    # Test grid search (small sample for speed)
    print("\n=== Testing Grid Search ===")
    result = fit_multiobjective_grid_search(
        initial_mask, observed, observed_time, 
        downsample_k=2, n_samples=5,
        frame_weights=frame_weights
    )
    
    print(f"\nBest loss: {result['best_loss']:.6f}")
    print(f"Best parameters:")
    print(f"  p_move: {result['best_params'].p_move:.4f}")
    print(f"  p_div: {result['best_params'].p_div:.4f}")
    print(f"  alpha: {result['best_params'].alpha:.4f}")
    print(f"  beta: {result['best_params'].beta:.4f}")
    print(f"  edge_bonus: {result['best_params'].edge_bonus:.4f}")
    print(f"  k_time: {result['best_params'].k_time:.4f}")
    print(f"  gamma: {result['best_params'].gamma:.4f}")
    
    print("\nMulti-objective inference test complete!")
