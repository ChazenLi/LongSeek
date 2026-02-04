import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from io_mat import load_binary_mask, load_measures
from preprocess.extract_observations import extract_frame_statistics
from typing import Dict, List
import json

def extract_all_observations(
    manifest_path: str = "CA_project/data_manifest.csv",
    mask_type: str = "gt_mask_path",
    use_measures: bool = True
) -> Dict[str, any]:
    """
    Extract observations from all frames.
    
    Args:
        manifest_path: Path to manifest CSV
        mask_type: Which mask type to use ('gt_mask_path', 'manual_mask_path', etc.)
        use_measures: Whether to use measures.mat data
        
    Returns:
        Dictionary with time series data
    """
    manifest = pd.read_csv(manifest_path)
    
    observations = {
        "frame_ids": [],
        "wound_area": [],
        "cell_coverage": [],
        "roughness": [],
        "time": [],
    }
    
    if use_measures:
        measures_path = Path("CA/DATA/SN15/SN15/measures.mat")
        measures_data = load_measures(measures_path)
        
        observations["measures_wound_area"] = []  # From measures.mat
        observations["measures_cell_pixels"] = []
    
    for idx, row in manifest.iterrows():
        frame_id = row["frame_id"]
        mask_path = row[mask_type]
        
        if pd.isna(mask_path) or mask_path == "":
            print(f"Skipping {frame_id}: no mask file")
            continue
        
        try:
            # Load binary mask
            mask = load_binary_mask(mask_path)
            
            # Extract statistics
            stats = extract_frame_statistics(mask)
            
            observations["frame_ids"].append(frame_id)
            observations["wound_area"].append(stats["wound_area"])
            observations["cell_coverage"].append(stats["cell_coverage"])
            observations["roughness"].append(stats["roughness"])
            
            # Use frame index as time proxy
            observations["time"].append(idx)
            
            # Add measures.mat data if available
            if use_measures:
                # Find corresponding frame in measures
                for frame_info in measures_data["frames"]:
                    if frame_info["name"] == frame_id:
                        # Estimate wound area from tp+tn+fp+fn
                        # Assuming tp+tn = cell pixels, fp+fn = wound pixels
                        # This is a rough estimate
                        total_pixels = frame_info["msc"]["tp"] + frame_info["msc"]["tn"] + \
                                     frame_info["msc"]["fp"] + frame_info["msc"]["fn"]
                        wound_pixels = frame_info["msc"]["fp"] + frame_info["msc"]["fn"]
                        cell_pixels = frame_info["msc"]["tp"] + frame_info["msc"]["tn"]
                        
                        observations["measures_wound_area"].append(wound_pixels)
                        observations["measures_cell_pixels"].append(cell_pixels)
                        break
        
        except Exception as e:
            print(f"Error processing {frame_id}: {e}")
            continue
    
    # Convert to numpy arrays
    for key in ["wound_area", "cell_coverage", "roughness", "time"]:
        observations[key] = np.array(observations[key])
    
    if use_measures and observations["measures_wound_area"]:
        observations["measures_wound_area"] = np.array(observations["measures_wound_area"])
        observations["measures_cell_pixels"] = np.array(observations["measures_cell_pixels"])
    
    return observations

def save_observations(observations: Dict[str, any], output_path: str):
    """Save observations to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    obs_serializable = {}
    for key, value in observations.items():
        if isinstance(value, np.ndarray):
            obs_serializable[key] = value.tolist()
        else:
            obs_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(obs_serializable, f, indent=2)
    
    print(f"Saved observations to {output_path}")

def load_observations(input_path: str) -> Dict[str, any]:
    """Load observations from JSON file."""
    import json
    with open(input_path, 'r') as f:
        obs = json.load(f)
    
    # Convert back to numpy arrays
    for key, value in obs.items():
        if isinstance(value, list):
            obs[key] = np.array(value)
    
    return obs

def create_comparison_dataset():
    """
    Create a comparison dataset comparing different mask sources.
    
    This is useful for the "multi-source consistency" experiment.
    """
    mask_types = ["gt_mask_path", "manual_mask_path", "multicellseg_path", "topman_path", "tscratch_path"]
    
    comparison = {}
    
    for mask_type in mask_types:
        print(f"\nExtracting observations from {mask_type}...")
        try:
            obs = extract_all_observations(mask_type=mask_type, use_measures=False)
            if len(obs["wound_area"]) > 0:
                comparison[mask_type] = obs
                print(f"  Extracted {len(obs['wound_area'])} frames")
        except Exception as e:
            print(f"  Error: {e}")
    
    return comparison

if __name__ == "__main__":
    # Extract observations from ground truth (reannotation)
    print("Extracting observations from reannotation masks...")
    observations = extract_all_observations(mask_type="gt_mask_path", use_measures=True)
    
    print(f"\nExtracted {len(observations['wound_area'])} frames")
    print(f"Time range: {observations['time'][0]} to {observations['time'][-1]}")
    print(f"Wound area range: {observations['wound_area'].min():.0f} to {observations['wound_area'].max():.0f}")
    
    # Save to file
    output_path = Path("CA_project/observations_reannotation.json")
    save_observations(observations, str(output_path))
    
    # Create comparison dataset
    print("\n\nCreating comparison dataset...")
    comparison = create_comparison_dataset()
    
    # Save comparison
    comparison_output = Path("CA_project/comparison_dataset.json")
    obs_serializable = {}
    for key, value in comparison.items():
        obs_serializable[key] = {}
        for k2, v2 in value.items():
            if isinstance(v2, np.ndarray):
                obs_serializable[key][k2] = v2.tolist()
            else:
                obs_serializable[key][k2] = v2
    
    with open(comparison_output, 'w') as f:
        json.dump(obs_serializable, f, indent=2)
    
    print(f"\nSaved comparison dataset to {comparison_output}")
    
    # Print summary
    print("\n\n=== Summary ===")
    for mask_type, obs in comparison.items():
        if len(obs["wound_area"]) > 0:
            print(f"{mask_type}:")
            print(f"  Frames: {len(obs['wound_area'])}")
            print(f"  Initial wound area: {obs['wound_area'][0]:.0f}")
            print(f"  Final wound area: {obs['wound_area'][-1]:.0f}")
            print(f"  Closure rate: {(obs['wound_area'][0] - obs['wound_area'][-1]) / len(obs['wound_area']):.2f} pixels/frame")
