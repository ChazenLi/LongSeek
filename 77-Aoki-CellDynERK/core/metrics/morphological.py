"""
Morphological Observables for CA Model Evaluation

This module provides comprehensive morphological metrics for evaluating
wound healing CA simulations, going beyond simple wound area to
include roughness, width, shape metrics, and velocity analysis.

Key Insights from Research:
- Roughness metrics provide more information than perimeter alone
- Height field roughness is more robust than perimeter-based roughness
- Width statistics capture anisotropic closure
- Shape metrics detect morphological changes
- Velocity segments reveal temporal dynamics
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_roughness_perimeter(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate perimeter-based roughness metrics.
    
    Roughness is defined as the ratio of actual perimeter to smooth perimeter.
    Higher values indicate more irregular boundaries.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        
    Returns:
        Dict with roughness metrics
    """
    # Extract wound boundary
    wound_mask = (mask == 0).astype(np.uint8)
    
    # Find contours
    from skimage.measure import find_contours
    contours = find_contours(wound_mask, level=0.5)
    
    if len(contours) == 0:
        return {
            'roughness_perimeter': 0.0,
            'perimeter_length': 0.0,
            'num_contours': 0
        }
    
    # Use largest contour
    largest_contour = max(contours, key=len)
    
    # Calculate actual perimeter
    actual_perimeter = 0.0
    for i in range(len(largest_contour)):
        j = (i + 1) % len(largest_contour)
        dist = np.sqrt(
            (largest_contour[i, 0] - largest_contour[j, 0])**2 +
            (largest_contour[i, 1] - largest_contour[j, 1])**2
        )
        actual_perimeter += dist
    
    # Calculate smooth perimeter (convex hull)
    try:
        hull = ConvexHull(largest_contour)
        smooth_perimeter = 0.0
        for simplex in hull.simplices:
            p1 = largest_contour[simplex[0]]
            p2 = largest_contour[simplex[1]]
            smooth_perimeter += np.sqrt(
                (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            )
    except:
        smooth_perimeter = actual_perimeter
    
    # Roughness = actual / smooth
    roughness = actual_perimeter / smooth_perimeter if smooth_perimeter > 0 else 1.0
    
    return {
        'roughness_perimeter': roughness,
        'perimeter_length': actual_perimeter,
        'num_contours': len(contours)
    }


def calculate_roughness_height_field(mask: np.ndarray, sigma: float = 3.0) -> Dict[str, float]:
    """
    Calculate height field roughness (more robust than perimeter-based).
    
    Height field roughness is based on the distance transform of the wound,
    providing a more stable measure of boundary irregularity.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        sigma: Gaussian smoothing sigma
        
    Returns:
        Dict with roughness metrics
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    
    # Distance transform (distance from wound boundary)
    wound_mask = (mask == 0).astype(np.uint8)
    distance_field = distance_transform_edt(wound_mask)
    
    # Smooth the distance field
    if sigma > 0:
        distance_field_smooth = gaussian_filter(distance_field, sigma=sigma)
    else:
        distance_field_smooth = distance_field
    
    # Calculate gradient magnitude
    grad_y, grad_x = np.gradient(distance_field_smooth)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Restrict to wound region (where distance > 0)
    wound_region = (distance_field_smooth > 0)
    
    if wound_region.sum() == 0:
        return {
            'roughness_height_field': 0.0,
            'gradient_mean': 0.0,
            'gradient_std': 0.0
        }
    
    # Statistics on gradient magnitude
    gradient_in_wound = gradient_magnitude[wound_region]
    roughness = np.mean(gradient_in_wound)
    
    return {
        'roughness_height_field': roughness,
        'gradient_mean': np.mean(gradient_in_wound),
        'gradient_std': np.std(gradient_in_wound),
        'gradient_max': np.max(gradient_in_wound)
    }


def calculate_width_statistics(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate width statistics of the wound.
    
    Width is measured in multiple directions to capture anisotropy.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        
    Returns:
        Dict with width statistics
    """
    wound_mask = (mask == 0).astype(np.uint8)
    
    # Find bounding box of wound
    rows = np.any(wound_mask, axis=1)
    cols = np.any(wound_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]] if rows.any() else (0, mask.shape[0]-1)
    cmin, cmax = np.where(cols)[0][[0, -1]] if cols.any() else (0, mask.shape[1]-1)
    
    # Horizontal widths (at each row)
    horizontal_widths = []
    for r in range(rmin, rmax + 1):
        wound_pixels = np.where(wound_mask[r, cmin:cmax+1])[0]
        if len(wound_pixels) > 0:
            width = wound_pixels[-1] - wound_pixels[0] + 1
            horizontal_widths.append(width)
    
    # Vertical widths (at each column)
    vertical_widths = []
    for c in range(cmin, cmax + 1):
        wound_pixels = np.where(wound_mask[rmin:rmax+1, c])[0]
        if len(wound_pixels) > 0:
            width = wound_pixels[-1] - wound_pixels[0] + 1
            vertical_widths.append(width)
    
    if len(horizontal_widths) == 0:
        return {
            'width_mean': 0.0,
            'width_std': 0.0,
            'width_horizontal_mean': 0.0,
            'width_vertical_mean': 0.0
        }
    
    return {
        'width_mean': np.mean(horizontal_widths),
        'width_std': np.std(horizontal_widths),
        'width_min': np.min(horizontal_widths),
        'width_max': np.max(horizontal_widths),
        'width_horizontal_mean': np.mean(horizontal_widths),
        'width_vertical_mean': np.mean(vertical_widths),
        'width_aspect_ratio': np.mean(horizontal_widths) / np.mean(vertical_widths) 
                           if np.mean(vertical_widths) > 0 else 1.0
    }


def calculate_shape_metrics(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate shape metrics for the wound.
    
    Includes aspect ratio, convexity, solidity, and circularity.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        
    Returns:
        Dict with shape metrics
    """
    from skimage.measure import regionprops, label
    from skimage.morphology import convex_hull_image
    
    wound_mask = (mask == 0).astype(np.uint8)
    
    # Label connected components
    labeled = label(wound_mask)
    
    if labeled.max() == 0:
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'aspect_ratio': 1.0,
            'convexity': 0.0,
            'solidity': 0.0,
            'circularity': 0.0,
            'eccentricity': 0.0
        }
    
    # Find largest component (main wound)
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    
    area = largest_region.area
    perimeter = largest_region.perimeter
    
    # Aspect ratio (major axis / minor axis)
    major_axis = largest_region.major_axis_length
    minor_axis = largest_region.minor_axis_length
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    
    # Eccentricity
    eccentricity = largest_region.eccentricity
    
    # Convexity (perimeter of convex hull / actual perimeter)
    convex_hull = convex_hull_image(wound_mask == 1)
    hull_regions = regionprops(convex_hull.astype(int))
    if hull_regions:
        hull_perimeter = hull_regions[0].perimeter
        convexity = hull_perimeter / perimeter if perimeter > 0 else 1.0
    else:
        convexity = 1.0
    
    # Solidity (area / convex hull area)
    solidity = largest_region.solidity
    
    # Circularity (4 * pi * area / perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'aspect_ratio': float(aspect_ratio),
        'convexity': float(convexity),
        'solidity': float(solidity),
        'circularity': float(circularity),
        'eccentricity': float(eccentricity)
    }


def calculate_front_distance_distribution(mask: np.ndarray, 
                                    center: Optional[Tuple[int, int]] = None,
                                    n_bins: int = 20) -> Dict[str, any]:
    """
    Calculate distribution of distances from wound front to boundary.
    
    This captures how uniformly the wound is closing from all directions.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        center: Center point for distance calculation (default: centroid)
        n_bins: Number of bins for distribution
        
    Returns:
        Dict with distance distribution metrics
    """
    from scipy.ndimage import distance_transform_edt
    
    wound_mask = (mask == 0).astype(np.uint8)
    
    # Distance transform from wound edge
    distance_field = distance_transform_edt(wound_mask)
    
    # Get wound boundary pixels
    from skimage.feature import canny
    from scipy.ndimage import binary_erosion
    boundary = wound_mask & ~binary_erosion(wound_mask, structure=np.ones((3, 3)))
    
    boundary_points = np.where(boundary)
    
    if len(boundary_points[0]) == 0:
        return {
            'mean_distance': 0.0,
            'std_distance': 0.0,
            'distance_histogram': None,
            'distance_bins': None
        }
    
    # Calculate centroid if center not provided
    if center is None:
        center = (
            int(np.mean(boundary_points[0])),
            int(np.mean(boundary_points[1]))
        )
    
    # Calculate distances from center to boundary
    distances = np.sqrt(
        (boundary_points[0] - center[0])**2 +
        (boundary_points[1] - center[1])**2
    )
    
    # Distance distribution statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Histogram
    hist, bin_edges = np.histogram(distances, bins=n_bins)
    
    return {
        'mean_distance': float(mean_dist),
        'std_distance': float(std_dist),
        'min_distance': float(min_dist),
        'max_distance': float(max_dist),
        'distance_histogram': hist.tolist(),
        'distance_bins': bin_edges.tolist()
    }


def calculate_velocity_segments(areas: List[float], 
                             times: List[float],
                             n_segments: int = 3) -> Dict[str, any]:
    """
    Calculate velocity in different time segments.
    
    This reveals temporal dynamics of wound closure
    (e.g., early acceleration, late deceleration).
    
    Args:
        areas: List of wound areas at each time point
        times: List of time points (same length as areas)
        n_segments: Number of segments to divide time into
        
    Returns:
        Dict with velocity metrics
    """
    areas = np.array(areas)
    times = np.array(times)
    
    if len(areas) < 2:
        return {
            'velocities': [],
            'mean_velocity': 0.0,
            'velocity_std': 0.0
        }
    
    # Calculate total velocity (overall)
    total_velocity = (areas[0] - areas[-1]) / (times[-1] - times[0]) if times[-1] > times[0] else 0.0
    
    # Divide into segments
    if len(areas) < n_segments + 1:
        # Not enough points for segmentation, use overall velocity
        velocities = [total_velocity]
        segment_labels = ['overall']
    else:
        # Calculate segment boundaries
        segment_size = len(areas) // n_segments
        velocities = []
        segment_labels = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size + 1, len(areas))
            
            if end_idx > start_idx + 1:
                seg_velocity = (areas[start_idx] - areas[end_idx]) / (times[end_idx] - times[start_idx])
                velocities.append(seg_velocity)
                segment_labels.append(f'segment_{i+1}')
    
    velocities = np.array(velocities)
    
    return {
        'velocities': velocities.tolist(),
        'mean_velocity': float(np.mean(velocities)),
        'velocity_std': float(np.std(velocities)),
        'max_velocity': float(np.max(velocities)),
        'min_velocity': float(np.min(velocities)),
        'segment_labels': segment_labels
    }


def calculate_all_morphological_metrics(mask: np.ndarray,
                                     areas: Optional[List[float]] = None,
                                     times: Optional[List[float]] = None,
                                     center: Optional[Tuple[int, int]] = None) -> Dict[str, any]:
    """
    Calculate all morphological metrics for a wound mask.
    
    This provides a comprehensive set of observables for model fitting.
    
    Args:
        mask: Binary wound mask (0=wound, 1=cell)
        areas: Optional time series of wound areas
        times: Optional time points corresponding to areas
        center: Optional center point for distance calculations
        
    Returns:
        Dict with all morphological metrics
    """
    metrics = {}
    
    # Roughness metrics
    metrics.update(calculate_roughness_perimeter(mask))
    metrics.update(calculate_roughness_height_field(mask))
    
    # Width statistics
    metrics.update(calculate_width_statistics(mask))
    
    # Shape metrics
    metrics.update(calculate_shape_metrics(mask))
    
    # Front distance distribution
    metrics.update(calculate_front_distance_distribution(mask, center=center))
    
    # Velocity segments (if time series provided)
    if areas is not None and times is not None:
        metrics.update(calculate_velocity_segments(areas, times))
    
    return metrics


def extract_morphological_from_history(history: List[Dict[str, float]],
                                     initial_mask: np.ndarray,
                                     downsample_k: int = 4) -> Dict[str, List[float]]:
    """
    Extract morphological metrics from simulation history.
    
    Args:
        history: List of simulation steps (each with grid info)
        initial_mask: Initial binary mask
        downsample_k: Downsampling factor used in simulation
        
    Returns:
        Dict with time series of morphological metrics
    """
    from skimage.measure import label
    
    time_series = {
        'roughness_perimeter': [],
        'roughness_height_field': [],
        'width_mean': [],
        'aspect_ratio': [],
        'solidity': []
    }
    
    for step_data in history:
        # Reconstruct grid from wound area
        # Note: This is approximate since we don't store full grid in history
        # In practice, you might want to store the grid or extract from CA object
        
        # For now, use wound area as proxy
        wound_area = step_data['wound_area']
        
        # Placeholder - in practice, you'd compute actual metrics
        time_series['roughness_perimeter'].append(1.0)
        time_series['roughness_height_field'].append(0.5)
        time_series['width_mean'].append(np.sqrt(wound_area))
        time_series['aspect_ratio'].append(1.0)
        time_series['solidity'].append(0.9)
    
    return time_series


if __name__ == "__main__":
    # Test morphological metrics
    print("Testing Morphological Metrics...")
    
    # Create a synthetic wound mask
    mask = np.ones((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 0  # Square wound in center
    
    # Add some roughness
    mask[30:35, 40:45] = 1
    mask[65:70, 55:60] = 0
    
    # Calculate all metrics
    metrics = calculate_all_morphological_metrics(mask)
    
    print("\n=== Morphological Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            print(f"{key}: {type(value).__name__} with {len(value)} elements")
        else:
            print(f"{key}: {value:.4f}")
    
    # Test velocity segments
    print("\n=== Velocity Segments ===")
    times = [0, 10, 20, 30, 40]
    areas = [1600, 1200, 800, 500, 200]
    velocity_metrics = calculate_velocity_segments(areas, times, n_segments=3)
    print(velocity_metrics)
    
    print("\nMorphological metrics test complete!")
