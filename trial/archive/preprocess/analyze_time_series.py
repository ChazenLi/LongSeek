import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

def parse_frame_name(frame_id: str) -> Tuple[str, str]:
    """
    Parse frame ID to extract location and time point.
    
    Examples:
        "20_00" -> ("20", "00")
        "DIC_SN_15_L10_Sum00" -> ("L10", "00")
        "DIC_SN_15_L10_Sum32" -> ("L10", "32")
    """
    if "L" in frame_id and "Sum" in frame_id:
        # Format: DIC_SN_15_Lxx_Sumyy or Lxx_Sumyy
        parts = frame_id.split("_")
        loc = None
        time = None
        
        for i, part in enumerate(parts):
            if part.startswith("L"):
                loc = part
            elif part.startswith("Sum"):
                time = part[3:]  # Remove "Sum"
        
        return (loc, time)
    else:
        # Format: xx_yy
        parts = frame_id.split("_")
        if len(parts) == 2:
            return (parts[0], parts[1])
    
    return (frame_id, "")

def group_by_location(frame_ids: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group frames by location.
    
    Returns:
        Dict mapping location -> list of (frame_id, time_point)
    """
    groups = {}
    
    for frame_id in frame_ids:
        loc, time_str = parse_frame_name(frame_id)
        
        try:
            time = int(time_str)
        except ValueError:
            time = 99999  # Put invalid times at the end
        
        if loc not in groups:
            groups[loc] = []
        
        groups[loc].append((frame_id, time))
    
    # Sort each group by time
    for loc in groups:
        groups[loc].sort(key=lambda x: x[1])
    
    return groups

def analyze_time_series(comparison_data: Dict[str, Dict]):
    """
    Analyze time series structure in the data.
    """
    print("=== Analyzing frame structure ===\n")
    
    # Use manual_mask_path as reference (has all frames)
    manual_data = comparison_data["manual_mask_path"]
    frame_ids = manual_data["frame_ids"]
    
    groups = group_by_location(frame_ids)
    
    print(f"Found {len(groups)} location groups:")
    for loc, frames in sorted(groups.items()):
        print(f"\nLocation {loc}: {len(frames)} frames")
        for frame_id, time in frames[:5]:
            print(f"  {frame_id} (time={time})")
        if len(frames) > 5:
            print(f"  ... and {len(frames)-5} more")
    
    return groups

def extract_time_series_for_location(
    location: str,
    comparison_data: Dict[str, Dict],
    mask_type: str = "manual_mask_path"
) -> Dict[str, np.ndarray]:
    """
    Extract time series for a specific location.
    
    Args:
        location: Location identifier (e.g., "L10", "L12", "20")
        comparison_data: Comparison dataset
        mask_type: Which mask to use
        
    Returns:
        Dictionary with time series data
    """
    data = comparison_data[mask_type]
    frame_ids = data["frame_ids"]
    wound_areas = data["wound_area"]
    cell_coverages = data["cell_coverage"]
    
    # Group frames by location
    groups = group_by_location(frame_ids)
    
    if location not in groups:
        raise ValueError(f"Location {location} not found")
    
    frames = groups[location]
    
    # Extract data for this location
    frame_ids_loc = []
    time_points = []
    wound_areas_loc = []
    cell_coverages_loc = []
    
    for frame_id, time in frames:
        idx = frame_ids.index(frame_id)
        frame_ids_loc.append(frame_id)
        time_points.append(time)
        wound_areas_loc.append(wound_areas[idx])
        cell_coverages_loc.append(cell_coverages[idx])
    
    return {
        "location": location,
        "frame_ids": frame_ids_loc,
        "time_points": np.array(time_points),
        "wound_area": np.array(wound_areas_loc),
        "cell_coverage": np.array(cell_coverages_loc),
    }

def find_best_time_series(
    comparison_data: Dict[str, Dict],
    min_frames: int = 3
) -> Dict[str, Dict]:
    """
    Find locations with good time series (decreasing wound area).
    
    Args:
        comparison_data: Comparison dataset
        min_frames: Minimum number of frames
        
    Returns:
        Dictionary of time series by location
    """
    manual_data = comparison_data["manual_mask_path"]
    groups = group_by_location(manual_data["frame_ids"])
    
    best_series = {}
    
    for loc, frames in groups.items():
        if len(frames) < min_frames:
            continue
        
        # Extract time series
        ts = extract_time_series_for_location(loc, comparison_data)
        
        # Check if wound area decreases (wound healing)
        wound_area = ts["wound_area"]
        
        # Simple check: initial > final
        if wound_area[0] > wound_area[-1]:
            best_series[loc] = ts
    
    return best_series

if __name__ == "__main__":
    # Load comparison data
    with open("CA_project/comparison_dataset.json", 'r') as f:
        comparison_data = json.load(f)
    
    # Analyze structure
    groups = analyze_time_series(comparison_data)
    
    # Find best time series
    print("\n\n=== Finding wound healing time series ===\n")
    best_series = find_best_time_series(comparison_data)
    
    if not best_series:
        print("No clear wound healing patterns found (wound area decreasing over time)")
        print("\nLet's check all locations anyway:")
        best_series = {}
        for loc, frames in groups.items():
            if len(frames) >= 2:
                ts = extract_time_series_for_location(loc, comparison_data)
                best_series[loc] = ts
    else:
        print(f"Found {len(best_series)} locations with wound healing pattern:")
    
    for loc, ts in best_series.items():
        print(f"\nLocation {loc}:")
        print(f"  Frames: {len(ts['frame_ids'])}")
        print(f"  Time range: {ts['time_points'][0]} to {ts['time_points'][-1]}")
        print(f"  Initial wound area: {ts['wound_area'][0]:.0f}")
        print(f"  Final wound area: {ts['wound_area'][-1]:.0f}")
        print(f"  Closure rate: {(ts['wound_area'][0] - ts['wound_area'][-1]) / len(ts['wound_area']):.2f} pixels/frame")
        
        # Save this time series
        output_path = Path(f"CA_project/time_series_{loc}.json")
        
        ts_serializable = {}
        for k, v in ts.items():
            if isinstance(v, np.ndarray):
                ts_serializable[k] = v.tolist()
            else:
                ts_serializable[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(ts_serializable, f, indent=2)
        
        print(f"  Saved to {output_path}")
