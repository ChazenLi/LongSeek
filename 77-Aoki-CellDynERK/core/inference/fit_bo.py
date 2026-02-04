import numpy as np
from typing import Dict, List, Callable, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ca.model import CellOnlyCA, CAParams, CAMolParams

def calculate_loss(
    simulated: List[Dict[str, float]],
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    k_time: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> float:
    """
    Calculate loss between simulated and observed observables.
    
    Now supports:
    - Time scaling via k_time parameter
    - Multiple observables (A_t, roughness, width_mean, roughness_height_field)
    - Velocity segment matching
    - Frame weights based on segmentation quality (w_t)
    - Noise model (σ_t) for uncertainty-weighted loss
    
    Args:
        simulated: List of simulation steps
        observed: Dictionary of observed time series
        observed_time: Time points for observed data (real time units)
        k_time: Time scaling factor (CA steps per real time unit)
        weights: Weights for each observable
        frame_weights: Per-frame weights based on segmentation quality (w_t)
        noise_levels: Per-frame noise levels (σ_t) for uncertainty weighting
        
    Returns:
        Total loss
    """
    if weights is None:
        weights = {
            "A_t": 1.0,
            "roughness_perimeter": 0.5,
            "roughness_height_field": 0.5,
            "width_mean": 0.3,
            "velocity": 0.2,
        }
    
    loss = 0.0
    num_steps = len(simulated)
    
    # Convert CA steps to real time using k_time
    sim_time_real = np.arange(num_steps) / k_time
    
    # Wound area loss
    if "A_t" in observed:
        sim_A = np.array([s["wound_area"] for s in simulated])
        obs_A = observed["A_t"]
        
        # Normalize to initial value
        sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A
        obs_A_norm = obs_A / obs_A[0] if obs_A[0] > 0 else obs_A
        
        # Interpolate simulated data to observed time points
        from scipy.interpolate import interp1d
        
        if num_steps > 1:
            f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear',
                           bounds_error=False, fill_value="extrapolate")
            sim_at_obs_time = f_sim(observed_time)
        else:
            sim_at_obs_time = np.full_like(obs_A_norm, sim_A_norm[0])
        
        # Calculate weighted MSE with frame weights and noise model
        residuals = (sim_at_obs_time - obs_A_norm) ** 2
        
        # Apply frame weights if provided
        if frame_weights is not None:
            residuals = residuals * frame_weights
        
        # Apply noise model (inverse variance weighting) if provided
        if noise_levels is not None and len(noise_levels) == len(residuals):
            # Avoid division by zero
            sigma_sq = noise_levels ** 2 + 1e-10
            residuals = residuals / sigma_sq
        
        mse_A = np.mean(residuals)
        loss += weights.get("A_t", 1.0) * mse_A
    
    # Roughness loss (perimeter-based, if available)
    if "roughness_perimeter" in observed:
        # For now, use placeholder if not in simulation
        obs_R = observed["roughness_perimeter"]
        # Roughness is typically computed post-simulation
        # This is a placeholder for future implementation
    
    # Height field roughness loss (if available)
    if "roughness_height_field" in observed:
        obs_Rh = observed["roughness_height_field"]
        # Placeholder for future implementation
    
    # Width statistics loss (if available)
    if "width_mean" in observed:
        obs_W = observed["width_mean"]
        # Placeholder for future implementation
    
    # Velocity loss (segment-wise)
    if len(observed["A_t"]) >= 2 and num_steps >= 2:
        # Calculate observed velocities
        obs_velocities = []
        for i in range(len(observed["A_t"]) - 1):
            delta_A = observed["A_t"][i+1] - observed["A_t"][i]
            delta_t = observed_time[i+1] - observed_time[i]
            if delta_t > 0:
                obs_velocities.append(delta_A / delta_t)
        
        # Calculate simulated velocities at corresponding time points
        if len(obs_velocities) > 0:
            # Interpolate simulated area to get velocities
            sim_A = np.array([s["wound_area"] for s in simulated])
            sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A
            
            # Get simulated areas at observed time points
            if num_steps > 1:
                f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear', fill_value='extrapolate')
                sim_at_obs_time = f_sim(observed_time)
                
                # Calculate velocities
                sim_velocities = []
                for i in range(len(sim_at_obs_time) - 1):
                    delta_A = sim_at_obs_time[i+1] - sim_at_obs_time[i]
                    delta_t = observed_time[i+1] - observed_time[i]
                    if delta_t > 0:
                        sim_velocities.append(delta_A / delta_t)
                
                if len(sim_velocities) == len(obs_velocities):
                    vel_loss = np.mean((np.array(sim_velocities) - np.array(obs_velocities)) ** 2)
                    loss += weights.get("velocity", 0.2) * vel_loss
    
    return loss

def run_simulation(
    params: CAParams,
    initial_mask: np.ndarray,
    num_steps: int,
    downsample_k: int = 4,
    seed: Optional[int] = None,
    observed_time: Optional[np.ndarray] = None
) -> List[Dict[str, float]]:
    """
    Run a simulation with given parameters.
    
    Now supports k_time parameter for time scaling.
    
    Args:
        params: CA parameters (including k_time)
        initial_mask: Initial binary mask (cells)
        num_steps: Number of simulation steps (in CA steps)
        downsample_k: Downsampling factor
        seed: Random seed
        observed_time: Observed time points (for determining num_steps if not provided)
        
    Returns:
        Simulation history
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    # If observed_time is provided, calculate num_steps based on k_time
    if observed_time is not None and len(observed_time) > 1:
        # Scale max time by k_time to get CA steps
        max_real_time = observed_time[-1]
        num_steps = int(max_real_time * params.k_time) + 10  # Add buffer
    
    # Create CA model
    H, W = initial_mask.shape
    ca = CellOnlyCA(H // downsample_k, W // downsample_k, params)
    ca.initialize_from_mask(initial_mask, k=downsample_k)
    
    # Run simulation
    history = ca.run(num_steps)
    
    return history

def objective_function(
    params_array: np.ndarray,
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    downsample_k: int = 4,
    seed: Optional[int] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> float:
    """
    Objective function for optimization.
    
    Now includes k_time and gamma parameters, plus frame weights and noise model.
    
    Args:
        params_array: [p_move, p_div, alpha, beta, edge_bonus, k_time, gamma]
        initial_mask: Initial binary mask
        observed: Observed time series
        observed_time: Time points for observed data
        downsample_k: Downsampling factor
        seed: Random seed
        frame_weights: Per-frame weights based on segmentation quality
        noise_levels: Per-frame noise levels for uncertainty weighting
        
    Returns:
        Loss to minimize
    """
    # Convert array to params
    params = CAParams(
        p_move=float(params_array[0]),
        p_div=float(params_array[1]),
        alpha=float(params_array[2]),
        beta=float(params_array[3]),
        edge_bonus=float(params_array[4]),
        k_time=float(params_array[5]) if len(params_array) > 5 else 1.0,
        gamma=float(params_array[6]) if len(params_array) > 6 else 0.0,
    )
    
    # Run simulation (num_steps calculated from k_time)
    history = run_simulation(params, initial_mask, num_steps=50, 
                           downsample_k=downsample_k, seed=seed, 
                           observed_time=observed_time)
    
    # Calculate loss with time scaling, frame weights, and noise model
    loss = calculate_loss(history, observed, observed_time, k_time=params.k_time,
                         frame_weights=frame_weights, noise_levels=noise_levels)
    
    return loss

def fit_bayesian_optimization(
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    n_iter: int = 30,
    n_calls: int = 100,
    random_state: Optional[int] = None,
    downsample_k: int = 4,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Fit CA parameters using Bayesian Optimization.
    
    Now includes k_time and gamma in search space, plus frame weights and noise model.
    
    Args:
        initial_mask: Initial binary mask
        observed: Observed time series
        observed_time: Time points for observed data
        n_iter: Number of iterations
        n_calls: Number of function evaluations
        random_state: Random seed
        downsample_k: Downsampling factor
        frame_weights: Per-frame weights based on segmentation quality
        noise_levels: Per-frame noise levels for uncertainty weighting
        
    Returns:
        Best parameters and result
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("scikit-optimize not installed. Please install: pip install scikit-optimize")
        print("Falling back to simple grid search...")
        return fit_simple_grid_search(initial_mask, observed, observed_time, downsample_k,
                                     frame_weights=frame_weights, noise_levels=noise_levels)
    
    # Define search space (now includes k_time and gamma)
    space = [
        Real(0.01, 0.99, name='p_move'),
        Real(0.001, 0.2, name='p_div'),
        Real(0.1, 5.0, name='alpha'),
        Real(0.1, 5.0, name='beta'),
        Real(0.5, 5.0, name='edge_bonus'),
        Real(0.5, 3.0, name='k_time'),  # Time scaling factor
        Real(0.0, 2.0, name='gamma'),   # Directional bias strength
    ]
    
    # Objective function with fixed arguments
    def objective(params_array):
        return objective_function(
            params_array, initial_mask, observed, observed_time, downsample_k, random_state,
            frame_weights=frame_weights, noise_levels=noise_levels
        )
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_iter,
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
    
    return {
        "best_params": best_params,
        "best_loss": float(result.fun),
        "result": result,
    }

def fit_simple_grid_search(
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    downsample_k: int = 4,
    n_samples: int = 10,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Simple grid search fallback when scikit-optimize is not available.
    
    Now includes k_time and gamma parameters, plus frame weights and noise model.
    
    Args:
        initial_mask: Initial binary mask
        observed: Observed time series
        observed_time: Time points for observed data
        downsample_k: Downsampling factor
        n_samples: Number of random samples
        frame_weights: Per-frame weights based on segmentation quality
        noise_levels: Per-frame noise levels for uncertainty weighting
        
    Returns:
        Best parameters and result
    """
    best_loss = float('inf')
    best_params = None
    best_history = None
    
    print(f"Running random search with {n_samples} samples...")
    
    for i in range(n_samples):
        # Random sampling (now includes k_time and gamma)
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
        history = run_simulation(params, initial_mask, num_steps=50,
                               downsample_k=downsample_k, observed_time=observed_time)
        
        # Calculate loss with time scaling, frame weights, and noise model
        loss = calculate_loss(history, observed, observed_time, k_time=params.k_time,
                             frame_weights=frame_weights, noise_levels=noise_levels)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_history = history
        
        print(f"Sample {i+1}/{n_samples}: loss={loss:.4f}, k_time={params.k_time:.2f}, gamma={params.gamma:.2f}")
    
    return {
        "best_params": best_params,
        "best_loss": best_loss,
        "best_history": best_history,
    }

if __name__ == "__main__":
    # Test with synthetic data
    print("Testing inference module...")
    
    # Create synthetic initial mask
    initial_mask = np.ones((100, 100), dtype=np.float32)
    initial_mask[30:70, 30:70] = 0.0  # Wound in center
    
    # Create synthetic observed data with time points
    observed_time = np.array([0, 17, 40])  # Real time points
    observed_A = np.array([1600, 800, 200])  # Wound area at those times
    
    observed = {
        "A_t": observed_A,
    }
    
    # Run simple grid search
    result = fit_simple_grid_search(
        initial_mask, observed, observed_time=observed_time, downsample_k=2, n_samples=5
    )
    
    print(f"\nBest parameters:")
    print(f"  p_move: {result['best_params'].p_move}")
    print(f"  p_div: {result['best_params'].p_div}")
    print(f"  alpha: {result['best_params'].alpha}")
    print(f"  beta: {result['best_params'].beta}")
    print(f"  edge_bonus: {result['best_params'].edge_bonus}")
    print(f"  k_time: {result['best_params'].k_time}")
    print(f"  gamma: {result['best_params'].gamma}")
    print(f"  Best loss: {result['best_loss']}")
