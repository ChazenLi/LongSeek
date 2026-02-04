import numpy as np
from skimage.measure import regionprops_table, label
from skimage.morphology import binary_closing, remove_small_holes, remove_small_objects, disk
from scipy import ndimage as ndi
from typing import Dict, Tuple, Optional

def binary_to_wound_mask(mask: np.ndarray, min_hole: int = 2000, 
                         min_object_size: int = 500, morph_cleanup: bool = True) -> np.ndarray:
    """
    Convert binary cell mask to wound (empty space) mask with improved processing.
    
    Improvements:
    - Extract only the largest connected component (main wound area)
    - Remove small objects and holes
    - Optional morphological cleanup
    
    Args:
        mask: Binary mask (1=cell, 0=empty)
        min_hole: Minimum hole size to keep
        min_object_size: Minimum object size to keep (before selecting largest component)
        morph_cleanup: Whether to apply morphological cleanup operations
        
    Returns:
        Binary wound mask (1=wound, 0=cell)
    """
    cell = mask > 0
    empty = ~cell.astype(np.uint8)
    
    # Remove small objects and holes
    empty = remove_small_objects(empty, min_object_size)
    empty = remove_small_holes(empty, min_hole)
    
    # Extract only the largest connected component (main wound area)
    label_output = label(empty)
    
    # Handle different return types from label function
    if isinstance(label_output, tuple):
        labeled = label_output[0]
    else:
        labeled = label_output
    
    max_label = int(np.max(labeled))
    if max_label > 0:
        # Find the largest component using regionprops
        regions = regionprops_table(labeled.astype(int), properties=['label', 'area'])
        if len(regions['area']) > 0:
            largest_label = int(regions['label'][np.argmax(regions['area'])])
            empty = (labeled == largest_label).astype(np.uint8)
    
    # Morphological cleanup
    if morph_cleanup:
        # Binary closing to fill small gaps
        empty = binary_closing(empty, disk(2))
        
        # Additional closing to smooth boundaries
        empty = binary_closing(empty, disk(1))
    
    return empty.astype(np.float32)

def extract_frontline(wound_mask: np.ndarray) -> np.ndarray:
    """
    Extract the frontline (boundary between wound and cells).
    
    Args:
        wound_mask: Binary wound mask
        
    Returns:
        Frontline as binary mask
    """
    # Compute morphological gradient using scikit-image
    from scipy.ndimage import binary_erosion, binary_dilation
    eroded = binary_erosion(wound_mask.astype(bool))
    dilated = binary_dilation(wound_mask.astype(bool))
    gradient = dilated.astype(np.float32) - eroded.astype(np.float32)
    
    return gradient

def calculate_wound_area(wound_mask: np.ndarray) -> float:
    """Calculate wound area (number of pixels)."""
    return float(np.sum(wound_mask > 0))

def calculate_frontline_roughness(frontline_mask: np.ndarray, wound_mask: np.ndarray) -> float:
    """
    Calculate frontline roughness as ratio of actual boundary to convex hull perimeter.
    
    Args:
        frontline_mask: Frontline boundary mask
        wound_mask: Wound mask
        
    Returns:
        Roughness value (>1.0 means rough)
    """
    from skimage.measure import find_contours
    
    # Get contours
    contours = find_contours(wound_mask.astype(float), 0.5)
    
    if len(contours) == 0:
        return 0.0
    
    # Use the longest contour
    largest_contour = max(contours, key=len)
    
    # Actual perimeter (sum of distances between consecutive points)
    if len(largest_contour) < 2:
        return 0.0
    
    actual_perimeter = np.sum(np.sqrt(np.sum(np.diff(largest_contour, axis=0)**2, axis=1)))
    
    # Approximate convex hull perimeter (use convex hull from skimage)
    from skimage.morphology import convex_hull_image
    hull_mask = convex_hull_image(wound_mask.astype(bool))
    hull_contours = find_contours(hull_mask.astype(float), 0.5)
    
    if len(hull_contours) == 0 or len(hull_contours[0]) < 2:
        return 0.0
    
    hull_perimeter = np.sum(np.sqrt(np.sum(np.diff(hull_contours[0], axis=0)**2, axis=1)))
    
    if hull_perimeter == 0:
        return 0.0
    
    return actual_perimeter / hull_perimeter

def downsample_binary(mask: np.ndarray, k: int, thr: float = 0.5) -> np.ndarray:
    """
    Downsample binary mask by kxk blocks.
    
    Args:
        mask: Binary mask
        k: Downsampling factor
        thr: Threshold for occupied (fraction > thr)
        
    Returns:
        Downsampled binary mask
    """
    H, W = mask.shape
    H2, W2 = H // k, W // k
    
    # Reshape and compute mean over blocks
    m = mask[:H2*k, :W2*k].reshape(H2, k, W2, k)
    frac = m.mean(axis=(1, 3))
    
    return (frac > thr).astype(np.float32)

def calculate_local_density(mask: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Calculate local density using sliding window.
    
    Args:
        mask: Binary mask
        window_size: Size of the window (odd number)
        
    Returns:
        Density map (0-1)
    """
    from scipy.ndimage import convolve
    
    kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
    density = convolve(mask.astype(np.float32), kernel, mode='constant', cval=0.0)
    
    return density

def calculate_frontline_width_stats(wound_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate wound width statistics (frontal advance distance).
    
    For each row, calculate the wound width by finding leftmost and rightmost
    cell boundaries. This gives the distribution of wound widths.
    
    Args:
        wound_mask: Binary wound mask (1=wound, 0=cell)
        
    Returns:
        Dictionary with width statistics: mean, std, min, max
    """
    H, W = wound_mask.shape
    widths = []
    
    for i in range(H):
        row = wound_mask[i, :]
        if np.any(row > 0):
            wound_indices = np.where(row > 0)[0]
            width = wound_indices[-1] - wound_indices[0]
            widths.append(width)
    
    if len(widths) == 0:
        return {"width_mean": 0.0, "width_std": 0.0, "width_min": 0.0, "width_max": 0.0}
    
    return {
        "width_mean": float(np.mean(widths)),
        "width_std": float(np.std(widths)),
        "width_min": float(np.min(widths)),
        "width_max": float(np.max(widths)),
    }

def calculate_height_field_roughness(wound_mask: np.ndarray) -> float:
    """
    Calculate roughness using height field representation.
    
    Treat the frontline as a function h(y) where for each row y,
    h(y) is the x-coordinate of the wound boundary closest to cells.
    
    Roughness = std(h(y)), which measures how "jagged" the front is.
    This is more robust to holes than perimeter-based roughness.
    
    Args:
        wound_mask: Binary wound mask (1=wound, 0=cell)
        
    Returns:
        Height field roughness (standard deviation of frontier positions)
    """
    H, W = wound_mask.shape
    frontier_positions = []
    
    for i in range(H):
        row = wound_mask[i, :]
        if np.any(row > 0):
            wound_indices = np.where(row > 0)[0]
            # Take the leftmost wound boundary as the frontier
            frontier_positions.append(float(wound_indices[0]))
    
    if len(frontier_positions) == 0:
        return 0.0
    
    return float(np.std(frontier_positions))

def calculate_frontline_velocity_series(observed_areas: np.ndarray, observed_time: np.ndarray) -> Dict[str, float]:
    """
    Calculate velocity statistics from wound area time series.
    
    Computes segment-wise velocities to capture multi-stage dynamics.
    Even with only 2-3 time points, this gives constraints on early vs late dynamics.
    
    Args:
        observed_areas: Wound area at each time point (absolute values)
        observed_time: Time points for observations
        
    Returns:
        Dictionary with velocity statistics for each time segment
    """
    velocities = {}
    
    if len(observed_areas) < 2:
        return {"velocity_mean": 0.0, "velocity_std": 0.0}
    
    # Calculate velocities between consecutive time points
    area_velocities = []
    time_velocities = []
    
    for i in range(len(observed_areas) - 1):
        delta_A = observed_areas[i+1] - observed_areas[i]
        delta_t = observed_time[i+1] - observed_time[i]
        
        if delta_t > 0:
            v = delta_A / delta_t
            area_velocities.append(v)
            time_velocities.append(delta_t)
            
            velocities[f"velocity_{i}_{i+1}"] = float(v)
    
    if len(area_velocities) > 0:
        velocities["velocity_mean"] = float(np.mean(area_velocities))
        velocities["velocity_std"] = float(np.std(area_velocities))
    
    return velocities

def extract_frame_statistics(mask: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive statistics from a frame's binary mask.
    
    Now includes:
    - Basic area and coverage
    - Perimeter-based roughness (original)
    - Height-field roughness (more robust)
    - Frontline width statistics
    - (Velocity statistics computed across multiple frames separately)
    
    Args:
        mask: Binary mask (1=cell, 0=empty)
        
    Returns:
        Dictionary of statistics
    """
    wound_mask = binary_to_wound_mask(mask)
    frontline_mask = extract_frontline(wound_mask)
    
    width_stats = calculate_frontline_width_stats(wound_mask)
    height_roughness = calculate_height_field_roughness(wound_mask)
    
    stats = {
        "total_pixels": mask.size,
        "cell_pixels": float(np.sum(mask > 0)),
        "wound_pixels": float(np.sum(wound_mask > 0)),
        "cell_coverage": float(np.mean(mask > 0)),
        "wound_area": calculate_wound_area(wound_mask),
        "frontline_pixels": float(np.sum(frontline_mask > 0)),
        "roughness_perimeter": calculate_frontline_roughness(frontline_mask, wound_mask),
        "roughness_height_field": height_roughness,
    }
    
    stats.update(width_stats)
    
    return stats

if __name__ == "__main__":
    # Test with a simple mask
    test_mask = np.zeros((100, 100), dtype=np.float32)
    test_mask[50:70, 50:70] = 1.0  # Simple square of cells
    
    stats = extract_frame_statistics(test_mask)
    print("Test statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
