"""
Field Consistency Metrics for Molecular Field Integration

This module provides metrics for evaluating consistency between
CA simulation dynamics and molecular field observations.

Key Metrics:
- Velocity-field alignment: How well simulated cell velocities match field gradients
- Field-CA coupling: Correlation between field intensity and CA activity
- Directional consistency: Agreement between preferred migration directions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_velocity_field_alignment(ca_velocity: Dict[str, np.ndarray],
                                    field_gradient: Dict[str, np.ndarray],
                                    mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate alignment between CA velocity field and molecular field gradient.
    
    This measures how well the simulated cell migration follows
    the molecular field guidance (e.g., chemotaxis).
    
    Args:
        ca_velocity: Dict with 'u', 'v' components of CA velocity
        field_gradient: Dict with 'grad_x', 'grad_y' of molecular field
        mask: Optional mask to restrict analysis to certain regions
        
    Returns:
        Dict with alignment metrics
    """
    # Extract velocity components
    u_ca = ca_velocity.get('u', np.zeros_like(field_gradient['grad_x']))
    v_ca = ca_velocity.get('v', np.zeros_like(field_gradient['grad_y']))
    
    # Extract field gradient
    grad_x = field_gradient['grad_x']
    grad_y = field_gradient['grad_y']
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        u_ca = u_ca[mask]
        v_ca = v_ca[mask]
        grad_x = grad_x[mask]
        grad_y = grad_y[mask]
    
    # Normalize vectors to unit length
    def normalize_vectors(u, v):
        magnitude = np.sqrt(u**2 + v**2)
        magnitude = np.maximum(magnitude, 1e-10)
        return u / magnitude, v / magnitude
    
    u_ca_norm, v_ca_norm = normalize_vectors(u_ca, v_ca)
    grad_norm_x, grad_norm_y = normalize_vectors(grad_x, grad_y)
    
    # Calculate cosine similarity
    cosine_sim = u_ca_norm * grad_norm_x + v_ca_norm * grad_norm_y
    mean_alignment = np.mean(cosine_sim)
    
    # Calculate angular difference
    angle_ca = np.arctan2(v_ca, u_ca)
    angle_grad = np.arctan2(grad_y, grad_x)
    angle_diff = np.abs(angle_ca - angle_grad)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    mean_angle_diff = np.mean(angle_diff)
    
    # Calculate correlation coefficients
    correlation_u = np.corrcoef(u_ca.flatten(), grad_x.flatten())[0, 1] if len(u_ca) > 1 else 0.0
    correlation_v = np.corrcoef(v_ca.flatten(), grad_y.flatten())[0, 1] if len(v_ca) > 1 else 0.0
    
    return {
        'mean_cosine_similarity': float(mean_alignment),
        'mean_angle_difference': float(mean_angle_diff),
        'mean_angle_difference_degrees': float(np.degrees(mean_angle_diff)),
        'correlation_u': float(correlation_u if not np.isnan(correlation_u) else 0.0),
        'correlation_v': float(correlation_v if not np.isnan(correlation_v) else 0.0)
    }


def calculate_field_ca_coupling(field: np.ndarray,
                             ca_activity: np.ndarray,
                             mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate coupling between molecular field intensity and CA activity.
    
    This measures how well the molecular field correlates with
    regions of high cellular activity (migration/proliferation).
    
    Args:
        field: Molecular field intensity (T, H, W) or (H, W)
        ca_activity: CA activity measure (e.g., local migration count)
        mask: Optional mask to restrict analysis
        
    Returns:
        Dict with coupling metrics
    """
    # Ensure 2D arrays
    if field.ndim == 3:
        field = np.mean(field, axis=0)  # Average over time
    
    if ca_activity.ndim == 3:
        ca_activity = np.mean(ca_activity, axis=0)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        field = field[mask]
        ca_activity = ca_activity[mask]
    
    # Flatten arrays
    field_flat = field.flatten()
    ca_flat = ca_activity.flatten()
    
    # Calculate correlation
    correlation = np.corrcoef(field_flat, ca_flat)[0, 1] if len(field_flat) > 1 else 0.0
    correlation = float(correlation if not np.isnan(correlation) else 0.0)
    
    # Calculate mutual information approximation
    from scipy.stats import entropy
    hist_2d, _, _ = np.histogram2d(field_flat, ca_flat, bins=20)
    pxy = hist_2d / hist_2d.sum()
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    mi = entropy(px) + entropy(py) - entropy(pxy.flatten())
    
    return {
        'field_ca_correlation': correlation,
        'mutual_information': float(mi),
        'field_mean': float(np.mean(field)),
        'field_std': float(np.std(field)),
        'ca_activity_mean': float(np.mean(ca_activity)),
        'ca_activity_std': float(np.std(ca_activity))
    }


def calculate_directional_consistency(field_gradients: List[Dict[str, np.ndarray]],
                                  ca_directions: List[Dict[str, np.ndarray]],
                                  mask: Optional[np.ndarray] = None) -> Dict[str, any]:
    """
    Calculate consistency of directional preferences over time.
    
    This measures how stable the relationship between
    molecular field guidance and cell migration is over time.
    
    Args:
        field_gradients: List of field gradients at each time step
        ca_directions: List of CA direction preferences at each time step
        mask: Optional mask to restrict analysis
        
    Returns:
        Dict with directional consistency metrics
    """
    n_timesteps = len(field_gradients)
    
    if n_timesteps == 0:
        return {
            'mean_alignment_time_series': [],
            'std_alignment_time_series': [],
            'overall_alignment': 0.0
        }
    
    alignments = []
    
    for t in range(n_timesteps):
        grad = field_gradients[t]
        ca_dir = ca_directions[t]
        
        # Calculate alignment at this timestep
        alignment = calculate_velocity_field_alignment(ca_dir, grad, mask)
        alignments.append(alignment['mean_cosine_similarity'])
    
    alignments = np.array(alignments)
    
    return {
        'mean_alignment_time_series': alignments.tolist(),
        'mean_alignment': float(np.mean(alignments)),
        'std_alignment': float(np.std(alignments)),
        'alignment_trend': float(np.polyfit(range(len(alignments)), alignments, 1)[0]) if len(alignments) > 1 else 0.0
    }


def calculate_opticflow_agreement(ca_velocity: Dict[str, np.ndarray],
                                 opticflow: Dict[str, np.ndarray],
                                 mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate agreement between CA velocity and optical flow measurements.
    
    Optical flow provides direct velocity measurements, so this
    validates whether CA simulation captures real cell migration dynamics.
    
    Args:
        ca_velocity: Dict with 'u', 'v' components of CA velocity
        opticflow: Dict with 'u', 'v' components of optical flow
        mask: Optional mask to restrict analysis
        
    Returns:
        Dict with agreement metrics
    """
    # Extract velocity components
    u_ca = ca_velocity.get('u', np.zeros_like(opticflow['u']))
    v_ca = ca_velocity.get('v', np.zeros_like(opticflow['v']))
    
    u_flow = opticflow['u']
    v_flow = opticflow['v']
    magnitude_flow = opticflow.get('magnitude', np.sqrt(u_flow**2 + v_flow**2))
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        u_ca = u_ca[mask]
        v_ca = v_ca[mask]
        u_flow = u_flow[mask]
        v_flow = v_flow[mask]
        magnitude_flow = magnitude_flow[mask]
    
    # Calculate velocity magnitude error
    ca_magnitude = np.sqrt(u_ca**2 + v_ca**2)
    magnitude_error = np.abs(ca_magnitude - magnitude_flow)
    mean_magnitude_error = np.mean(magnitude_error)
    relative_magnitude_error = mean_magnitude_error / (np.mean(magnitude_flow) + 1e-10)
    
    # Calculate direction error
    ca_angle = np.arctan2(v_ca, u_ca)
    flow_angle = np.arctan2(v_flow, u_flow)
    angle_error = np.abs(ca_angle - flow_angle)
    angle_error = np.minimum(angle_error, 2*np.pi - angle_error)
    mean_angle_error = np.mean(angle_error)
    
    # Calculate endpoint error (Euclidean distance between velocity vectors)
    endpoint_error = np.sqrt((u_ca - u_flow)**2 + (v_ca - v_flow)**2)
    mean_endpoint_error = np.mean(endpoint_error)
    
    # Calculate correlation
    corr_u = np.corrcoef(u_ca.flatten(), u_flow.flatten())[0, 1] if len(u_ca) > 1 else 0.0
    corr_v = np.corrcoef(v_ca.flatten(), v_flow.flatten())[0, 1] if len(v_ca) > 1 else 0.0
    
    return {
        'mean_magnitude_error': float(mean_magnitude_error),
        'relative_magnitude_error': float(relative_magnitude_error),
        'mean_angle_error_radians': float(mean_angle_error),
        'mean_angle_error_degrees': float(np.degrees(mean_angle_error)),
        'mean_endpoint_error': float(mean_endpoint_error),
        'correlation_u': float(corr_u if not np.isnan(corr_u) else 0.0),
        'correlation_v': float(corr_v if not np.isnan(corr_v) else 0.0)
    }


def calculate_field_guidance_effectiveness(field: np.ndarray,
                                       ca_trajectories: np.ndarray,
                                       initial_positions: np.ndarray,
                                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate how effectively the field guides cell migration.
    
    This measures whether cells tend to move in directions
    consistent with the field gradient.
    
    Args:
        field: Molecular field (H, W)
        ca_trajectories: Cell trajectories (N_cells, 2, N_steps)
        initial_positions: Initial cell positions (N_cells, 2)
        mask: Optional mask to restrict analysis
        
    Returns:
        Dict with guidance effectiveness metrics
    """
    from scipy.ndimage import gradient
    
    # Calculate field gradient
    grad_y, grad_x = np.gradient(field)
    
    # For each cell, calculate displacement
    n_cells, _, n_steps = ca_trajectories.shape
    guidance_scores = []
    
    for cell in range(n_cells):
        for step in range(1, n_steps):
            # Current position
            y, x = ca_trajectories[cell, :, step-1].astype(int)
            
            # Check bounds
            if 0 <= y < field.shape[0] and 0 <= x < field.shape[1]:
                # Field gradient at this position
                field_grad_x = grad_x[y, x]
                field_grad_y = grad_y[y, x]
                
                # Actual cell displacement
                dy = ca_trajectories[cell, 0, step] - ca_trajectories[cell, 0, step-1]
                dx = ca_trajectories[cell, 1, step] - ca_trajectories[cell, 1, step-1]
                
                # Calculate alignment
                field_mag = np.sqrt(field_grad_x**2 + field_grad_y**2)
                cell_mag = np.sqrt(dx**2 + dy**2)
                
                if field_mag > 0 and cell_mag > 0:
                    alignment = (dx * field_grad_x + dy * field_grad_y) / (field_mag * cell_mag)
                    guidance_scores.append(alignment)
    
    if len(guidance_scores) == 0:
        return {
            'mean_guidance_score': 0.0,
            'fraction_guided': 0.0
        }
    
    guidance_scores = np.array(guidance_scores)
    
    # Fraction of movements that are aligned (cosine similarity > 0.5)
    fraction_guided = np.mean(guidance_scores > 0.5)
    
    return {
        'mean_guidance_score': float(np.mean(guidance_scores)),
        'std_guidance_score': float(np.std(guidance_scores)),
        'fraction_guided': float(fraction_guided)
    }


if __name__ == "__main__":
    # Test field consistency metrics
    print("Testing Field Consistency Metrics...")
    
    # Create synthetic test data
    H, W = 50, 50
    
    # Synthetic molecular field (simple gradient)
    field = np.linspace(0, 1, H)[:, np.newaxis] * np.ones((1, W))
    
    # Synthetic CA velocity (aligned with field)
    u_ca = np.zeros((H, W)) + 0.1
    v_ca = np.ones((H, W)) * 0.5
    
    ca_velocity = {'u': u_ca, 'v': v_ca}
    
    # Field gradient
    grad_y, grad_x = np.gradient(field)
    field_gradient = {'grad_x': grad_x, 'grad_y': grad_y}
    
    # Test velocity-field alignment
    alignment = calculate_velocity_field_alignment(ca_velocity, field_gradient)
    print("\n=== Velocity-Field Alignment ===")
    for key, value in alignment.items():
        print(f"{key}: {value:.4f}")
    
    # Test field-CA coupling
    ca_activity = np.random.random((H, W))
    coupling = calculate_field_ca_coupling(field, ca_activity)
    print("\n=== Field-CA Coupling ===")
    for key, value in coupling.items():
        print(f"{key}: {value:.4f}")
    
    # Test opticflow agreement
    opticflow = {
        'u': u_ca + 0.05,  # Slightly different
        'v': v_ca + 0.05,
        'magnitude': np.sqrt((u_ca + 0.05)**2 + (v_ca + 0.05)**2)
    }
    agreement = calculate_opticflow_agreement(ca_velocity, opticflow)
    print("\n=== Opticflow Agreement ===")
    for key, value in agreement.items():
        print(f"{key}: {value:.4f}")
    
    print("\nField consistency metrics test complete!")
