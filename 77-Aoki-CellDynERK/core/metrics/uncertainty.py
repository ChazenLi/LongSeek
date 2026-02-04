"""
Uncertainty Quantification for CA Model Parameters

This module provides methods for quantifying uncertainty in
parameter estimation, including:
- Parameter correlation analysis
- Confidence intervals
- Bootstrapping methods
- Bayesian posterior analysis

Key Concepts:
- Identifiability: Can parameters be uniquely determined from data?
- Uncertainty: How confident are we in parameter estimates?
- Correlation: Which parameters trade off against each other?
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def calculate_parameter_correlation(params_samples: np.ndarray,
                                 param_names: List[str],
                                 threshold: float = 0.7) -> Dict[str, any]:
    """
    Calculate correlation matrix between parameters from optimization samples.
    
    High correlation (> threshold) indicates parameters are not
    independently identifiable (trade off against each other).
    
    Args:
        params_samples: Array of parameter samples (n_samples, n_params)
        param_names: Names of parameters
        threshold: Correlation threshold for high correlation
        
    Returns:
        Dict with correlation matrix and highly correlated pairs
    """
    n_samples, n_params = params_samples.shape
    
    if n_samples < 2:
        return {
            'correlation_matrix': np.eye(n_params),
            'highly_correlated_pairs': [],
            'identifiability_score': 1.0
        }
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(params_samples.T)
    
    # Find highly correlated pairs (above threshold in absolute value)
    highly_correlated = []
    
    for i in range(n_params):
        for j in range(i+1, n_params):
            corr = abs(corr_matrix[i, j])
            if corr > threshold:
                highly_correlated.append({
                    'param1': param_names[i],
                    'param2': param_names[j],
                    'correlation': float(corr)
                })
    
    # Identifiability score (fraction of pairs with low correlation)
    n_pairs = n_params * (n_params - 1) / 2
    n_low_corr = n_pairs - len(highly_correlated)
    identifiability_score = n_low_corr / n_pairs if n_pairs > 0 else 1.0
    
    return {
        'correlation_matrix': corr_matrix.tolist(),
        'highly_correlated_pairs': highly_correlated,
        'identifiability_score': float(identifiability_score),
        'num_correlated_pairs': len(highly_correlated),
        'total_pairs': int(n_pairs)
    }


def calculate_confidence_intervals(params_samples: np.ndarray,
                                 param_names: List[str],
                                 confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals for parameters from samples.
    
    Args:
        params_samples: Array of parameter samples (n_samples, n_params)
        param_names: Names of parameters
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Dict with confidence intervals for each parameter
    """
    n_samples, n_params = params_samples.shape
    
    if n_samples < 2:
        return {name: {'lower': 0.0, 'upper': 0.0, 'median': 0.0}
                for name in param_names}
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_dict = {}
    
    for i, name in enumerate(param_names):
        samples = params_samples[:, i]
        
        ci_dict[name] = {
            'lower': float(np.percentile(samples, lower_percentile)),
            'upper': float(np.percentile(samples, upper_percentile)),
            'median': float(np.median(samples)),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples))
        }
    
    return ci_dict


def bootstrap_parameter_uncertainty(loss_function: Callable,
                                  param_initial: np.ndarray,
                                  n_bootstrap: int = 100,
                                  n_samples_per_fit: int = 30,
                                  confidence_level: float = 0.95,
                                  **fit_kwargs) -> Dict[str, any]:
    """
    Bootstrap parameter uncertainty by resampling observed data.
    
    Args:
        loss_function: Function that calculates loss for given parameters
        param_initial: Initial parameter guess
        n_bootstrap: Number of bootstrap samples
        n_samples_per_fit: Number of samples per bootstrap fit
        confidence_level: Confidence level for intervals
        **fit_kwargs: Additional arguments for fitting
        
    Returns:
        Dict with bootstrap results
    """
    # Collect bootstrap parameter estimates
    bootstrap_params = []
    bootstrap_losses = []
    
    for i in range(n_bootstrap):
        # Fit parameters on resampled data
        # Note: In practice, you'd resample observations
        # For now, we just use random samples from parameter space
        
        # This is a simplified bootstrap
        # In practice, you'd:
        # 1. Resample observed data with replacement
        # 2. Fit parameters to resampled data
        # 3. Store parameter estimates
        
        # Placeholder: sample from parameter space
        n_params = len(param_initial)
        sample = param_initial + np.random.randn(n_params) * 0.1 * param_initial
        loss = loss_function(sample)
        
        bootstrap_params.append(sample)
        bootstrap_losses.append(loss)
    
    bootstrap_params = np.array(bootstrap_params)
    bootstrap_losses = np.array(bootstrap_losses)
    
    # Calculate confidence intervals
    param_names = [f'param_{i}' for i in range(len(param_initial))]
    confidence_intervals = calculate_confidence_intervals(
        bootstrap_params, param_names, confidence_level
    )
    
    return {
        'bootstrap_samples': bootstrap_params.tolist(),
        'bootstrap_losses': bootstrap_losses.tolist(),
        'confidence_intervals': confidence_intervals,
        'median_loss': float(np.median(bootstrap_losses)),
        'best_loss': float(np.min(bootstrap_losses))
    }


def calculate_parameter_sensitivity(loss_function: Callable,
                                  param_base: np.ndarray,
                                  param_ranges: List[Tuple[float, float]],
                                  n_points: int = 50) -> Dict[str, any]:
    """
    Calculate sensitivity of loss to parameter changes.
    
    This shows which parameters have most impact on model fit.
    
    Args:
        loss_function: Function that calculates loss for given parameters
        param_base: Base parameter values
        param_ranges: List of (min, max) for each parameter
        n_points: Number of points to sample for each parameter
        
    Returns:
        Dict with sensitivity analysis results
    """
    base_loss = loss_function(param_base)
    n_params = len(param_base)
    
    sensitivity_results = {}
    
    for i in range(n_params):
        # Vary parameter i while keeping others at base value
        param_min, param_max = param_ranges[i]
        param_values = np.linspace(param_min, param_max, n_points)
        
        losses = []
        for value in param_values:
            params_test = param_base.copy()
            params_test[i] = value
            loss = loss_function(params_test)
            losses.append(loss)
        
        losses = np.array(losses)
        
        # Calculate sensitivity metrics
        sensitivity_results[f'param_{i}'] = {
            'param_values': param_values.tolist(),
            'loss_values': losses.tolist(),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses)),
            'loss_range': float(np.max(losses) - np.min(losses)),
            'base_loss': float(base_loss),
            'optimal_value': float(param_values[np.argmin(losses)]),
            'sensitivity_score': float(np.max(losses) - np.min(losses)) / (np.max(losses) + 1e-10)
        }
    
    # Sort by sensitivity score
    sorted_params = sorted(sensitivity_results.items(),
                         key=lambda x: x[1]['sensitivity_score'],
                         reverse=True)
    
    return {
        'sensitivity_results': sensitivity_results,
        'sorted_sensitivity': [(k, v['sensitivity_score']) for k, v in sorted_params],
        'base_loss': float(base_loss)
    }


def calculate_information_criteria(loss_values: List[float],
                                n_params: int,
                                n_data_points: int) -> Dict[str, float]:
    """
    Calculate information criteria for model comparison.
    
    - AIC (Akaike Information Criterion): penalizes complexity
    - BIC (Bayesian Information Criterion): stronger penalty for complexity
    - Lower values indicate better models
    
    Args:
        loss_values: List of loss values from different models/fits
        n_params: Number of parameters in model
        n_data_points: Number of data points
        
    Returns:
        Dict with AIC and BIC values
    """
    loss_array = np.array(loss_values)
    
    # AIC = n * ln(RSS/n) + 2k
    # where RSS is residual sum of squares, k is number of parameters
    # Note: For MSE-based loss, RSS = n * MSE
    aic = n_data_points * np.log(loss_array) + 2 * n_params
    
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = n_data_points * np.log(loss_array) + n_params * np.log(n_data_points)
    
    # Calculate Akaike weights (model probabilities)
    min_aic = np.min(aic)
    delta_aic = aic - min_aic
    aic_weights = np.exp(-0.5 * delta_aic) / np.sum(np.exp(-0.5 * delta_aic))
    
    # Calculate BIC weights
    min_bic = np.min(bic)
    delta_bic = bic - min_bic
    bic_weights = np.exp(-0.5 * delta_bic) / np.sum(np.exp(-0.5 * delta_bic))
    
    return {
        'aic': aic.tolist(),
        'bic': bic.tolist(),
        'aic_weights': aic_weights.tolist(),
        'bic_weights': bic_weights.tolist(),
        'best_aic_idx': int(np.argmin(aic)),
        'best_bic_idx': int(np.argmin(bic))
    }


def assess_identifiability(correlation_matrix: np.ndarray,
                          param_names: List[str],
                          threshold: float = 0.8) -> Dict[str, any]:
    """
    Assess parameter identifiability from correlation matrix.
    
    A parameter is identifiable if:
    1. Low correlation with other parameters
    2. Narrow confidence intervals
    3. Consistent across different fitting methods
    
    Args:
        correlation_matrix: Parameter correlation matrix (n_params, n_params)
        param_names: Names of parameters
        threshold: Correlation threshold for identifiability
        
    Returns:
        Dict with identifiability assessment
    """
    n_params = len(param_names)
    
    # For each parameter, check correlations with others
    param_identifiable = {}
    
    for i, name in enumerate(param_names):
        correlations_with_others = [abs(correlation_matrix[i, j]) 
                                    for j in range(n_params) if i != j]
        
        max_correlation = max(correlations_with_others) if correlations_with_others else 0.0
        mean_correlation = np.mean(correlations_with_others) if correlations_with_others else 0.0
        
        # Parameter is identifiable if max correlation < threshold
        is_identifiable = max_correlation < threshold
        
        param_identifiable[name] = {
            'identifiable': is_identifiable,
            'max_correlation': float(max_correlation),
            'mean_correlation': float(mean_correlation),
            'correlation_with': [
                {'param': param_names[j], 'correlation': float(correlation_matrix[i, j])}
                for j in range(n_params) if i != j
            ]
        }
    
    # Overall identifiability
    n_identifiable = sum(1 for p in param_identifiable.values() if p['identifiable'])
    identifiability_fraction = n_identifiable / n_params
    
    return {
        'param_identifiability': param_identifiable,
        'identifiability_fraction': float(identifiability_fraction),
        'n_identifiable': n_identifiable,
        'n_total': n_params,
        'assessment': (
            'High' if identifiability_fraction > 0.7 else
            'Medium' if identifiability_fraction > 0.4 else
            'Low'
        )
    }


def plot_parameter_correlation(params_samples: np.ndarray,
                               param_names: List[str],
                               save_path: Optional[str] = None):
    """
    Plot parameter correlation matrix.
    
    Args:
        params_samples: Array of parameter samples (n_samples, n_params)
        param_names: Names of parameters
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(params_samples.T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot correlation matrix
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Add correlation values
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black')
    
    ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test uncertainty quantification
    print("Testing Uncertainty Quantification...")
    
    # Generate synthetic parameter samples
    np.random.seed(42)
    n_samples = 100
    n_params = 7
    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma']
    
    # Create correlated samples (simulating real parameter distributions)
    base_samples = np.random.randn(n_samples, n_params)
    
    # Add some correlations
    base_samples[:, 0] += 0.5 * base_samples[:, 2]  # p_move correlated with alpha
    base_samples[:, 1] += 0.3 * base_samples[:, 3]  # p_div correlated with beta
    
    # Scale to realistic ranges
    params_samples = np.zeros_like(base_samples)
    params_samples[:, 0] = 0.85 + 0.1 * base_samples[:, 0]  # p_move
    params_samples[:, 1] = 0.17 + 0.05 * base_samples[:, 1]  # p_div
    params_samples[:, 2] = 1.45 + 0.2 * base_samples[:, 2]  # alpha
    params_samples[:, 3] = 0.70 + 0.1 * base_samples[:, 3]  # beta
    params_samples[:, 4] = 1.41 + 0.2 * base_samples[:, 4]  # edge_bonus
    params_samples[:, 5] = 1.5 + 0.3 * base_samples[:, 5]  # k_time
    params_samples[:, 6] = 0.0 + 0.1 * base_samples[:, 6]  # gamma
    
    # Test parameter correlation
    print("\n=== Parameter Correlation ===")
    corr_analysis = calculate_parameter_correlation(params_samples, param_names)
    print(f"Identifiability score: {corr_analysis['identifiability_score']:.4f}")
    print(f"Highly correlated pairs: {len(corr_analysis['highly_correlated_pairs'])}")
    for pair in corr_analysis['highly_correlated_pairs']:
        print(f"  {pair['param1']} - {pair['param2']}: {pair['correlation']:.3f}")
    
    # Test confidence intervals
    print("\n=== Confidence Intervals ===")
    ci_dict = calculate_confidence_intervals(params_samples, param_names, confidence_level=0.95)
    for name, ci in ci_dict.items():
        print(f"{name}: [{ci['lower']:.4f}, {ci['upper']:.4f}], median={ci['median']:.4f}")
    
    # Test identifiability assessment
    print("\n=== Identifiability Assessment ===")
    identifiability = assess_identifiability(
        np.array(corr_analysis['correlation_matrix']),
        param_names,
        threshold=0.8
    )
    print(f"Overall identifiability: {identifiability['assessment']}")
    print(f"Identifiable parameters: {identifiability['n_identifiable']}/{identifiability['n_total']}")
    
    # Test information criteria
    print("\n=== Information Criteria ===")
    loss_values = [0.0042, 0.0050, 0.0038, 0.0060]
    ic = calculate_information_criteria(loss_values, n_params=7, n_data_points=3)
    print(f"AIC: {ic['aic']}")
    print(f"BIC: {ic['bic']}")
    print(f"Best model (AIC): idx {ic['best_aic_idx']}")
    print(f"Best model (BIC): idx {ic['best_bic_idx']}")
    
    # Plot correlation
    print("\n=== Plotting Correlation ===")
    fig = plot_parameter_correlation(params_samples, param_names)
    plt.savefig('results/parameter_correlation_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved correlation plot to results/parameter_correlation_test.png")
    
    print("\nUncertainty quantification test complete!")
