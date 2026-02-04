"""
Universal Data Loader Interface for CA Framework

This module provides a standardized interface for loading wound healing data
from different sources and formats, making the framework dataset-agnostic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class DatasetLoader:
    """
    Base class for dataset loaders.
    
    Subclasses should implement load_observations() and load_initial_mask().
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
    
    def load_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load observation time points and wound areas.
        
        Returns:
            (observed_time, observed_wound_area)
        """
        raise NotImplementedError("Subclasses must implement load_observations()")
    
    def load_initial_mask(self) -> np.ndarray:
        """
        Load initial binary mask (cells=1, empty=0).
        
        Returns:
            Initial mask as numpy array
        """
        raise NotImplementedError("Subclasses must implement load_initial_mask()")
    
    def get_metadata(self) -> Dict:
        """Get dataset metadata."""
        return {
            "dataset_name": self.dataset_name,
            "loader_class": self.__class__.__name__,
        }


class JSONTimeSeriesLoader(DatasetLoader):
    """
    Load time series data from JSON file.
    
    Expected JSON format:
    {
        "location": "L12",
        "frame_ids": ["frame1", "frame2", "frame3"],
        "time_points": [0, 17, 40],
        "wound_area": [494292.0, 382946.0, 176100.0]
    }
    """
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        super().__init__(self.data.get('location', 'unknown'))
    
    def load_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        observed_time = np.array(self.data["time_points"])
        observed_wound_area = np.array(self.data["wound_area"])
        return observed_time, observed_wound_area
    
    def get_metadata(self) -> Dict:
        metadata = super().get_metadata()
        metadata.update({
            "location": self.data.get("location"),
            "frame_ids": self.data.get("frame_ids"),
            "n_timepoints": len(self.data["time_points"]),
        })
        return metadata


class MaskFileLoader:
    """
    Load mask from various file formats.
    """
    
    @staticmethod
    def load_from_mat(mat_path: str, var_name: str = 'manual_mask') -> np.ndarray:
        """Load mask from MATLAB .mat file."""
        try:
            from scipy.io import loadmat
            data = loadmat(mat_path)
            mask = data[var_name]
            return mask.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load .mat file: {e}")
    
    @staticmethod
    def load_from_image(img_path: str) -> np.ndarray:
        """Load mask from image file (PNG, TIF, etc.)."""
        try:
            from PIL import Image
            img = Image.open(img_path).convert('L')
            mask = np.array(img) > 127
            return mask.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load image file: {e}")
    
    @staticmethod
    def load_from_npy(npy_path: str) -> np.ndarray:
        """Load mask from .npy file."""
        mask = np.load(npy_path)
        return mask.astype(np.float32)


class UniversalDataLoader:
    """
    High-level interface for loading complete dataset for CA simulation.
    
    Usage:
        loader = UniversalDataLoader(
            time_series_json="time_series_L12.json",
            mask_file="mask.mat",
            mask_format="mat"
        )
        
        observed_time, observed_area = loader.load_observations()
        initial_mask = loader.load_initial_mask()
    """
    
    def __init__(
        self,
        time_series_json: Optional[str] = None,
        mask_file: Optional[str] = None,
        mask_format: str = "mat",
        time_points: Optional[List] = None,
        wound_areas: Optional[List] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            time_series_json: Path to JSON file with time series
            mask_file: Path to mask file
            mask_format: Format of mask file ('mat', 'image', 'npy')
            time_points: Direct time points (if no JSON)
            wound_areas: Direct wound areas (if no JSON)
        """
        self.time_series_json = time_series_json
        self.mask_file = mask_file
        self.mask_format = mask_format
        self._time_points = time_points
        self._wound_areas = wound_areas
        
        # Load metadata
        self.metadata = {}
        if time_series_json:
            loader = JSONTimeSeriesLoader(time_series_json)
            self.metadata = loader.get_metadata()
    
    def load_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load observation data.
        
        Returns:
            (observed_time, observed_wound_area)
        """
        if self.time_series_json:
            loader = JSONTimeSeriesLoader(self.time_series_json)
            return loader.load_observations()
        elif self._time_points and self._wound_areas:
            return np.array(self._time_points), np.array(self._wound_areas)
        else:
            raise ValueError("No observation data source specified")
    
    def load_initial_mask(self) -> np.ndarray:
        """Load initial mask."""
        if not self.mask_file:
            raise ValueError("No mask file specified")
        
        if self.mask_format == "mat":
            return MaskFileLoader.load_from_mat(self.mask_file)
        elif self.mask_format == "image":
            return MaskFileLoader.load_from_image(self.mask_file)
        elif self.mask_format == "npy":
            return MaskFileLoader.load_from_npy(self.mask_file)
        else:
            raise ValueError(f"Unknown mask format: {self.mask_format}")
    
    def get_observed_dict(self) -> Dict[str, np.ndarray]:
        """
        Get observed data in format expected by CA framework.
        
        Returns:
            {"A_t": wound_area_array}
        """
        _, observed_wound_area = self.load_observations()
        return {"A_t": observed_wound_area}


class MinimalSyntheticData:
    """
    Generate minimal synthetic dataset for validation and testing.
    
    This creates a simple wound healing scenario without requiring
    external data files.
    """
    
    @staticmethod
    def create_circular_wound(
        size: int = 128,
        wound_radius: int = 30,
        center: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create a circular wound mask.
        
        Args:
            size: Image size
            wound_radius: Radius of wound
            center: Wound center (default: image center)
            
        Returns:
            Binary mask (1=cell, 0=wound)
        """
        if center is None:
            center = (size // 2, size // 2)
        
        mask = np.ones((size, size), dtype=np.float32)
        
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        mask[dist_from_center < wound_radius] = 0.0
        
        return mask
    
    @staticmethod
    def create_synthetic_time_series(
        initial_area: float = 3000.0,
        time_points: List = [0, 17, 40],
        closure_rate: float = 0.65
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic wound area time series.
        
        Args:
            initial_area: Initial wound area
            time_points: Time points
            closure_rate: Fraction of wound closed by last time point
            
        Returns:
            (time_points, wound_areas)
        """
        time_points = np.array(time_points)
        
        # Exponential decay model
        t_norm = time_points / time_points[-1]
        wound_areas = initial_area * (1 - closure_rate * t_norm)
        
        return time_points, wound_areas


# Convenience functions for common workflows

def load_dataset_from_config(config: Dict) -> UniversalDataLoader:
    """
    Create data loader from configuration dictionary.
    
    Args:
        config: Configuration dict with keys:
            - time_series_json: path to JSON
            - mask_file: path to mask
            - mask_format: 'mat', 'image', or 'npy'
            
    Returns:
        UniversalDataLoader instance
    """
    return UniversalDataLoader(
        time_series_json=config.get("time_series_json"),
        mask_file=config.get("mask_file"),
        mask_format=config.get("mask_format", "mat"),
    )


def create_minimal_validation_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create minimal dataset for validation/testing.
    
    Returns:
        (initial_mask, observed_time, observed_area, metadata)
    """
    # Create synthetic mask
    initial_mask = MinimalSyntheticData.create_circular_wound()
    
    # Create synthetic time series
    observed_time, observed_area = MinimalSyntheticData.create_synthetic_time_series()
    
    metadata = {
        "dataset_type": "synthetic",
        "wound_geometry": "circular",
        "size": initial_mask.shape[0],
        "initial_area": float(np.sum(1 - initial_mask)),
    }
    
    return initial_mask, observed_time, observed_area, metadata


if __name__ == "__main__":
    # Test minimal dataset creation
    print("Testing minimal dataset creation...")
    
    mask, time_pts, areas, meta = create_minimal_validation_dataset()
    
    print(f"Mask shape: {mask.shape}")
    print(f"Initial wound area: {meta['initial_area']:.0f} pixels")
    print(f"Time points: {time_pts}")
    print(f"Wound areas: {areas}")
    print(f"\nMetadata: {meta}")
