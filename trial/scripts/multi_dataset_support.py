"""
Multi-dataset support module for CA modeling across different experimental conditions.

Supports:
- TScratch: Multiple locations (starve*, PC2*)
- MDCK: MDCK cell line data
- Melanoma: Melanoma cell line data
- Cross-location shared mechanisms with location-specific initial conditions
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_mat import load_binary_mask, load_measures, calculate_frame_weights, calculate_noise_model

class DatasetConfig:
    """Configuration for a specific dataset."""
    
    def __init__(self, name: str, root_path: str, condition: str = ""):
        self.name = name
        self.root_path = Path(root_path)
        self.condition = condition
        self.locations = []
        
    def get_location_path(self, location_id: str) -> Path:
        """Get the path for a specific location."""
        return self.root_path / location_id
    
    def list_available_locations(self) -> List[str]:
        """List all available locations in this dataset."""
        if not self.locations:
            if self.root_path.exists():
                self.locations = [d.name for d in self.root_path.iterdir() if d.is_dir()]
        return self.locations


class MultiDatasetManager:
    """
    Manager for handling multiple datasets with cross-location shared mechanisms.
    
    Key features:
    - Support multiple datasets (TScratch, MDCK, Melanoma)
    - Extract location-specific initial conditions
    - Enable cross-location parameter sharing
    - Integrate observations from multiple sources
    """
    
    def __init__(self):
        self.datasets = {}
        self.shared_params = None
        
    def register_dataset(self, config: DatasetConfig):
        """Register a new dataset."""
        self.datasets[config.name] = config
        
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        """Get a registered dataset by name."""
        return self.datasets.get(name)
    
    def list_all_locations(self, dataset_name: Optional[str] = None) -> Dict[str, List[str]]:
        """List all locations across all or specific datasets."""
        result = {}
        
        if dataset_name:
            if dataset_name in self.datasets:
                result[dataset_name] = self.datasets[dataset_name].list_available_locations()
        else:
            for name, config in self.datasets.items():
                result[name] = config.list_available_locations()
        
        return result
    
    def load_location_data(self, dataset_name: str, location_id: str, 
                          segmentation_sources: List[str] = None) -> Dict:
        """
        Load data for a specific location across all time points.
        
        Args:
            dataset_name: Name of the dataset
            location_id: Location identifier (e.g., "L12", "starve_1")
            segmentation_sources: List of segmentation sources to use 
                                 (e.g., ["manual", "multiCellSeg", "tscratch"])
                                 If None, use all available sources
        
        Returns:
            Dictionary containing masks, observations, and metadata
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not registered")
        
        config = self.datasets[dataset_name]
        location_path = config.get_location_path(location_id)
        
        if not location_path.exists():
            raise ValueError(f"Location {location_id} not found in dataset {dataset_name}")
        
        # Find all frames for this location
        frames = self._find_frames_for_location(location_path, location_id)
        
        # Load masks and extract observations
        observations = {}
        masks = {}
        
        for frame_info in frames:
            frame_id = frame_info['frame_id']
            
            # Load masks from different sources
            frame_masks = {}
            if segmentation_sources is None:
                segmentation_sources = ['manual', 'multiCellSeg', 'topman', 'tscratch']
            
            for source in segmentation_sources:
                mask_path = location_path / f"{source}_mat" / f"{frame_id}_{source}.mat"
                if mask_path.exists():
                    try:
                        mask = load_binary_mask(mask_path)
                        frame_masks[source] = mask
                    except Exception as e:
                        print(f"Warning: Failed to load {source} mask for {frame_id}: {e}")
            
            if frame_masks:
                masks[frame_id] = frame_masks
        
        # Load measures if available
        measures_path = location_path / "measures.mat"
        measures_data = None
        if measures_path.exists():
            try:
                measures_data = load_measures(measures_path)
            except Exception as e:
                print(f"Warning: Failed to load measures for {location_id}: {e}")
        
        return {
            'dataset': dataset_name,
            'location': location_id,
            'condition': config.condition,
            'frames': frames,
            'masks': masks,
            'measures': measures_data,
        }
    
    def _find_frames_for_location(self, location_path: Path, location_id: str) -> List[Dict]:
        """Find all frames for a given location."""
        frames = []
        
        # Look in images directory
        img_dir = location_path / "images"
        if img_dir.exists():
            img_files = sorted(img_dir.glob("*.tif"))
            for img_path in img_files:
                frame_id = img_path.stem
                frames.append({
                    'frame_id': frame_id,
                    'img_path': str(img_path),
                })
        
        return frames
    
    def extract_initial_condition(self, location_data: Dict, 
                                 source: str = "manual") -> np.ndarray:
        """
        Extract initial condition (first frame mask) for a location.
        
        Args:
            location_data: Location data dictionary from load_location_data
            source: Segmentation source to use
            
        Returns:
            Initial mask as numpy array
        """
        frames = location_data['frames']
        masks = location_data['masks']
        
        if not frames:
            raise ValueError("No frames found for this location")
        
        first_frame_id = frames[0]['frame_id']
        
        if first_frame_id not in masks:
            raise ValueError(f"No masks found for frame {first_frame_id}")
        
        if source not in masks[first_frame_id]:
            available = list(masks[first_frame_id].keys())
            raise ValueError(f"Source {source} not available. Available: {available}")
        
        return masks[first_frame_id][source]
    
    def extract_time_series(self, location_data: Dict, 
                           source: str = "manual") -> Dict[str, np.ndarray]:
        """
        Extract time series observations for a location.
        
        Args:
            location_data: Location data dictionary from load_location_data
            source: Segmentation source to use
            
        Returns:
            Dictionary of time series (A_t, time points, etc.)
        """
        from CA_project.preprocess.extract_observations import extract_frame_statistics
        
        frames = location_data['frames']
        masks = location_data['masks']
        
        time_points = []
        wound_areas = []
        coverages = []
        roughness = []
        
        for i, frame_info in enumerate(frames):
            frame_id = frame_info['frame_id']
            
            if frame_id not in masks or source not in masks[frame_id]:
                continue
            
            mask = masks[frame_id][source]
            stats = extract_frame_statistics(mask)
            
            time_points.append(i)  # Use frame index as time
            wound_areas.append(stats['wound_area'])
            coverages.append(stats['cell_coverage'])
            roughness.append(stats['roughness_perimeter'])
        
        return {
            'time': np.array(time_points, dtype=np.float32),
            'A_t': np.array(wound_areas, dtype=np.float32),
            'coverage': np.array(coverages, dtype=np.float32),
            'roughness': np.array(roughness, dtype=np.float32),
        }
    
    def fit_cross_location(self, 
                          dataset_name: str,
                          location_ids: List[str],
                          fit_function,
                          shared_params: bool = True) -> Dict:
        """
        Fit parameters across multiple locations with shared mechanisms.
        
        Args:
            dataset_name: Name of the dataset
            location_ids: List of location IDs to fit
            fit_function: Fitting function (e.g., fit_bayesian_optimization)
            shared_params: Whether to enforce shared parameters across locations
            
        Returns:
            Fitting results for each location
        """
        results = {}
        
        for location_id in location_ids:
            print(f"\n=== Fitting location {location_id} ===")
            
            # Load location data
            location_data = self.load_location_data(dataset_name, location_id)
            
            # Extract initial condition and time series
            try:
                initial_mask = self.extract_initial_condition(location_data)
                observations = self.extract_time_series(location_data)
                
                # Get frame weights and noise levels if measures available
                frame_weights = None
                noise_levels = None
                if location_data['measures']:
                    frame_weights = calculate_frame_weights(location_data['measures'])
                    noise_levels = calculate_noise_model(location_data['measures'])
                
                # Run fitting
                result = fit_function(
                    initial_mask=initial_mask,
                    observed=observations,
                    observed_time=observations['time'],
                    frame_weights=frame_weights,
                    noise_levels=noise_levels,
                )
                
                results[location_id] = {
                    'result': result,
                    'initial_mask': initial_mask,
                    'observations': observations,
                }
                
                print(f"Best loss for {location_id}: {result.get('best_loss', 'N/A')}")
                
            except Exception as e:
                print(f"Error fitting location {location_id}: {e}")
                results[location_id] = {'error': str(e)}
        
        return results


def register_tscratch_datasets(manager: MultiDatasetManager, base_path: str = "CA/DATA"):
    """
    Register TScratch datasets.
    
    TScratch has multiple conditions:
    - starve: Serum-starved conditions
    - PC2: With PC2 treatment
    """
    
    # Starve datasets
    for i in range(1, 5):  # Assume 4 locations
        config = DatasetConfig(
            name=f"tscratch_starve_{i}",
            root_path=f"{base_path}/TScratch/starve_{i}",
            condition="starve"
        )
        manager.register_dataset(config)
    
    # PC2 datasets
    for i in range(1, 5):  # Assume 4 locations
        config = DatasetConfig(
            name=f"tscratch_pc2_{i}",
            root_path=f"{base_path}/TScratch/PC2_{i}",
            condition="PC2"
        )
        manager.register_dataset(config)


def register_mdck_datasets(manager: MultiDatasetManager, base_path: str = "CA/DATA"):
    """Register MDCK datasets."""
    
    for i in range(1, 5):  # Assume 4 locations
        config = DatasetConfig(
            name=f"mdck_{i}",
            root_path=f"{base_path}/MDCK/location_{i}",
            condition="MDCK"
        )
        manager.register_dataset(config)


def register_melanoma_datasets(manager: MultiDatasetManager, base_path: str = "CA/DATA"):
    """Register Melanoma datasets."""
    
    for i in range(1, 5):  # Assume 4 locations
        config = DatasetConfig(
            name=f"melanoma_{i}",
            root_path=f"{base_path}/Melanoma/location_{i}",
            condition="melanoma"
        )
        manager.register_dataset(config)


if __name__ == "__main__":
    # Example usage
    manager = MultiDatasetManager()
    
    # Register SN15 dataset (original)
    sn15_config = DatasetConfig(
        name="sn15",
        root_path="CA/DATA/SN15/SN15",
        condition="SN15"
    )
    manager.register_dataset(sn15_config)
    
    # Register TScratch datasets
    register_tscratch_datasets(manager)
    
    # List all available locations
    all_locations = manager.list_all_locations()
    print("Available datasets and locations:")
    for dataset_name, locations in all_locations.items():
        print(f"  {dataset_name}: {len(locations)} locations")
        for loc in locations[:3]:  # Show first 3
            print(f"    - {loc}")
        if len(locations) > 3:
            print(f"    ... and {len(locations)-3} more")
    
    print("\nMulti-dataset support module loaded successfully!")
