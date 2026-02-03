"""
Field Preprocessing for Molecular Data

This module provides preprocessing functions for molecular field data
including background subtraction, normalization, gradient computation,
and field alignment for CA integration.

Functions:
- Background subtraction
- Quantile normalization  
- Gaussian smoothing
- Gradient computation
- Field interpolation and alignment
- Field quality assessment
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional, Union, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FieldPreprocessor:
    """
    Comprehensive field preprocessing for molecular data.
    
    This class provides a unified interface for preprocessing molecular
    field data to make it suitable for integration with CA models.
    """
    
    def __init__(self, 
                 background_percentile: float = 10.0,
                 norm_quantiles: Tuple[float, float] = (1.0, 99.0),
                 smoothing_sigma: float = 1.0,
                 gradient_method: str = "central_difference"):
        """
        Initialize field preprocessor.
        
        Args:
            background_percentile: Percentile for background estimation
            norm_quantiles: Quantile range for normalization
            smoothing_sigma: Gaussian smoothing sigma
            gradient_method: Method for gradient computation
        """
        self.background_percentile = background_percentile
        self.norm_quantiles = norm_quantiles
        self.smoothing_sigma = smoothing_sigma
        self.gradient_method = gradient_method
    
    def remove_background(self, field: np.ndarray, method: str = "percentile") -> np.ndarray:
        """
        Remove background from molecular field.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            method: Background estimation method ("percentile", "rolling_ball", "minimum")
            
        Returns:
            Background-subtracted field
        """
        if method == "percentile":
            # Use percentile-based background estimation
            if field.ndim == 3:
                # For 3D data, estimate background per timepoint
                bg_values = []
                for t in range(field.shape[0]):
                    bg_val = np.percentile(field[t], self.background_percentile)
                    bg_values.append(bg_val)
                background = np.array(bg_values)
                if background.ndim == 1:
                    background = background[:, np.newaxis, np.newaxis]
            else:
                background = np.percentile(field, self.background_percentile)
        
        elif method == "minimum":
            # Use minimum as background estimate
            if field.ndim == 3:
                background = np.min(field, axis=(1, 2), keepdims=True)
            else:
                background = np.min(field)
        
        elif method == "rolling_ball":
            # Rolling ball algorithm for background estimation
            from skimage.morphology import white_tophat, ball
            
            if field.ndim == 3:
                # Apply rolling ball to each timepoint
                background = np.zeros_like(field)
                for t in range(field.shape[0]):
                    selem = ball(radius=max(5, self.smoothing_sigma * 2))
                    background[t] = white_tophat(field[t], selem)
            else:
                selem = ball(radius=max(5, self.smoothing_sigma * 2))
                background = white_tophat(field, selem)
        
        else:
            raise ValueError(f"Unknown background method: {method}")
        
        # Subtract background
        field_corrected = field - background
        
        # Ensure non-negative values
        field_corrected = np.maximum(field_corrected, 0)
        
        logger.info(f"Background removed using {method} method")
        return field_corrected
    
    def normalize_field(self, field: np.ndarray, 
                       method: str = "quantile", 
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Normalize molecular field values.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            method: Normalization method ("quantile", "zscore", "minmax")
            target_range: Target value range for normalization
            
        Returns:
            Normalized field
        """
        if method == "quantile":
            # Quantile-based normalization to handle outliers
            q_low, q_high = self.norm_quantiles
            if field.ndim == 3:
                # Normalize each timepoint independently
                normalized = np.zeros_like(field, dtype=np.float32)
                for t in range(field.shape[0]):
                    q_vals = np.percentile(field[t], [q_low, q_high])
                    field_clipped = np.clip(field[t], q_vals[0], q_vals[1])
                    if q_vals[1] > q_vals[0]:
                        normalized[t] = (field_clipped - q_vals[0]) / (q_vals[1] - q_vals[0])
                        normalized[t] = normalized[t] * (target_range[1] - target_range[0]) + target_range[0]
            else:
                q_vals = np.percentile(field, [q_low, q_high])
                field_clipped = np.clip(field, q_vals[0], q_vals[1])
                if q_vals[1] > q_vals[0]:
                    normalized = (field_clipped - q_vals[0]) / (q_vals[1] - q_vals[0])
                    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
                else:
                    normalized = field
        
        elif method == "zscore":
            # Z-score normalization
            mean_val = np.mean(field)
            std_val = np.std(field)
            if std_val > 0:
                normalized = (field - mean_val) / std_val
            else:
                normalized = field - mean_val
            # Rescale to target range
            if np.std(normalized) > 0:
                normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        
        elif method == "minmax":
            # Min-max normalization
            min_val = np.min(field)
            max_val = np.max(field)
            if max_val > min_val:
                normalized = (field - min_val) / (max_val - min_val)
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            else:
                normalized = np.full_like(field, target_range[0])
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Field normalized using {method} method to range {target_range}")
        return normalized.astype(np.float32)
    
    def smooth_field(self, field: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply Gaussian smoothing to molecular field.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            sigma: Gaussian smoothing sigma (uses default if None)
            
        Returns:
            Smoothed field
        """
        if sigma is None:
            sigma = self.smoothing_sigma
        
        if field.ndim == 3:
            # Apply smoothing to each timepoint
            smoothed = np.zeros_like(field, dtype=np.float32)
            for t in range(field.shape[0]):
                smoothed[t] = ndimage.gaussian_filter(field[t], sigma=sigma)
        else:
            smoothed = ndimage.gaussian_filter(field, sigma=sigma)
        
        logger.info(f"Field smoothed with Gaussian sigma={sigma}")
        return smoothed
    
    def compute_gradients(self, field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute spatial gradients of molecular field.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            
        Returns:
            Dict with gradient components
        """
        if self.gradient_method == "central_difference":
            # Central difference gradients
            if field.ndim == 3:
                grad_x = np.zeros_like(field, dtype=np.float32)
                grad_y = np.zeros_like(field, dtype=np.float32)
                
                for t in range(field.shape[0]):
                    grad_x[t] = np.gradient(field[t], axis=1)
                    grad_y[t] = np.gradient(field[t], axis=0)
            else:
                grad_x = np.gradient(field, axis=1)
                grad_y = np.gradient(field, axis=0)
        
        elif self.gradient_method == "sobol":
            # Sobel filter for gradients
            if field.ndim == 3:
                grad_x = np.zeros_like(field, dtype=np.float32)
                grad_y = np.zeros_like(field, dtype=np.float32)
                
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
                
                for t in range(field.shape[0]):
                    grad_x[t] = ndimage.convolve(field[t], sobel_x)
                    grad_y[t] = ndimage.convolve(field[t], sobel_y)
            else:
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
                
                grad_x = ndimage.convolve(field, sobel_x)
                grad_y = ndimage.convolve(field, sobel_y)
        
        else:
            raise ValueError(f"Unknown gradient method: {self.gradient_method}")
        
        # Compute gradient magnitude and direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        gradients = {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_magnitude': grad_magnitude,
            'grad_direction': grad_direction
        }
        
        logger.info(f"Gradients computed using {self.gradient_method} method")
        return gradients
    
    def align_to_ca_grid(self, field: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Align molecular field to CA grid dimensions.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            target_shape: Target CA grid shape (H_ca, W_ca)
            
        Returns:
            Aligned field
        """
        if field.ndim == 2:
            h, w = field.shape
            target_h, target_w = target_shape
            
            # Compute scaling factors
            scale_h = target_h / h
            scale_w = target_w / w
            
            # Resize using zoom for better quality
            zoom_factors = (scale_h, scale_w)
            aligned = ndimage.zoom(field, zoom_factors, order=3)  # Cubic interpolation
            
        elif field.ndim == 3:
            t, h, w = field.shape
            target_h, target_w = target_shape
            
            scale_h = target_h / h
            scale_w = target_w / w
            
            zoom_factors = (1, scale_h, scale_w)  # Don't scale time dimension
            aligned = ndimage.zoom(field, zoom_factors, order=3)
        
        else:
            raise ValueError(f"Field must be 2D or 3D, got {field.ndim}D")
        
        # Ensure exact target dimensions
        if field.ndim == 2:
            aligned = aligned[:target_shape[0], :target_shape[1]]
        else:
            aligned = aligned[:, :target_shape[0], :target_shape[1]]
        
        logger.info(f"Field aligned from {field.shape} to target {target_shape}")
        return aligned
    
    def temporal_interpolate(self, field: np.ndarray, target_timepoints: np.ndarray, 
                           source_timepoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolate field to target time points.
        
        Args:
            field: Input field (T, H, W)
            target_timepoints: Target time points for interpolation
            source_timepoints: Source time points (uniform if None)
            
        Returns:
            Temporally interpolated field
        """
        if field.ndim != 3:
            raise ValueError("Temporal interpolation requires 3D field (T, H, W)")
        
        t, h, w = field.shape
        
        if source_timepoints is None:
            source_timepoints = np.linspace(0, 1, t)  # Normalize to [0, 1]
        
        # Normalize target timepoints
        target_timepoints_norm = (target_timepoints - target_timepoints[0]) / (target_timepoints[-1] - target_timepoints[0])
        
        # Interpolate each spatial point independently
        interpolated = np.zeros((len(target_timepoints_norm), h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                interpolated[:, i, j] = np.interp(
                    target_timepoints_norm, 
                    source_timepoints, 
                    field[:, i, j]
                )
        
        logger.info(f"Field interpolated from {t} to {len(target_timepoints)} time points")
        return interpolated
    
    def assess_field_quality(self, field: np.ndarray) -> Dict[str, float]:
        """
        Assess quality metrics for molecular field.
        
        Args:
            field: Input field (T, H, W) or (H, W)
            
        Returns:
            Dict with quality metrics
        """
        metrics = {}
        
        # Signal-to-noise ratio estimate
        signal = np.mean(field)
        noise = np.std(field)
        metrics['snr'] = signal / (noise + 1e-10)
        
        # Dynamic range
        metrics['dynamic_range'] = np.max(field) - np.min(field)
        
        # Coefficient of variation
        metrics['cv'] = np.std(field) / (np.mean(field) + 1e-10)
        
        # Spatial autocorrelation (only for 2D)
        if field.ndim == 2:
            from scipy.signal import correlate2d
            autocorr = correlate2d(field, field, mode='same')
            center = field.shape[0] // 2, field.shape[1] // 2
            metrics['spatial_autocorr'] = autocorr[center] / (np.var(field) + 1e-10)
        
        # Temporal variance (only for 3D)
        if field.ndim == 3:
            metrics['temporal_variance'] = np.mean(np.var(field, axis=0))
            metrics['temporal_cv'] = np.mean(np.std(field, axis=0) / (np.mean(field, axis=0) + 1e-10))
        
        logger.info(f"Field quality assessed: {len(metrics)} metrics computed")
        return metrics
    
    def process_field(self, field: np.ndarray, 
                     ca_shape: Optional[Tuple[int, int]] = None,
                     target_timepoints: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Complete field preprocessing pipeline.
        
        Args:
            field: Input molecular field
            ca_shape: Target CA grid shape for alignment
            target_timepoints: Target time points for temporal interpolation
            
        Returns:
            Dict with processed field components
        """
        logger.info("Starting complete field preprocessing pipeline")
        
        # Step 1: Background subtraction
        field_bg_corrected = self.remove_background(field)
        
        # Step 2: Normalization
        field_normalized = self.normalize_field(field_bg_corrected)
        
        # Step 3: Smoothing
        field_smoothed = self.smooth_field(field_normalized)
        
        # Step 4: Gradient computation
        gradients = self.compute_gradients(field_smoothed)
        
        # Step 5: Spatial alignment to CA grid
        if ca_shape is not None:
            field_aligned = self.align_to_ca_grid(field_smoothed, ca_shape)
            for key in gradients:
                gradients[key] = self.align_to_ca_grid(gradients[key], ca_shape)
        else:
            field_aligned = field_smoothed
        
        # Step 6: Temporal interpolation
        if target_timepoints is not None and field.ndim == 3:
            field_aligned = self.temporal_interpolate(field_aligned, target_timepoints)
            for key in gradients:
                if gradients[key].ndim == 3:
                    gradients[key] = self.temporal_interpolate(gradients[key], target_timepoints)
        
        # Step 7: Quality assessment
        quality_metrics = self.assess_field_quality(field_aligned)
        
        result = {
            'field': field_aligned,
            'gradients': gradients,
            'quality_metrics': quality_metrics
        }
        
        logger.info("Field preprocessing pipeline completed")
        return result


# Convenience functions for common preprocessing tasks

def preprocess_erk_field(erk_field: np.ndarray, ca_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Preprocess ERK molecular field for CA integration.
    
    Args:
        erk_field: ERK activity field (T, H, W)
        ca_shape: Target CA grid shape
        
    Returns:
        Preprocessed ERK field data
    """
    preprocessor = FieldPreprocessor(
        background_percentile=5.0,  # ERK has low background
        norm_quantiles=(1.0, 99.9),
        smoothing_sigma=2.0,
        gradient_method="central_difference"
    )
    
    return preprocessor.process_field(erk_field, ca_shape=ca_shape)


def preprocess_mlc_field(mlc_field: np.ndarray, ca_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Preprocess MLC molecular field for CA integration.
    
    Args:
        mlc_field: MLC activity field (T, H, W)
        ca_shape: Target CA grid shape
        
    Returns:
        Preprocessed MLC field data
    """
    preprocessor = FieldPreprocessor(
        background_percentile=10.0,  # MLC has moderate background
        norm_quantiles=(0.5, 99.5),
        smoothing_sigma=1.5,
        gradient_method="sobol"
    )
    
    return preprocessor.process_field(mlc_field, ca_shape=ca_shape)


def preprocess_optical_flow(flow_data: Dict[str, np.ndarray], ca_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Preprocess optical flow data for CA integration.
    
    Args:
        flow_data: Dict with 'u', 'v', 'magnitude', 'angle' components
        ca_shape: Target CA grid shape
        
    Returns:
        Preprocessed flow data
    """
    preprocessor = FieldPreprocessor(
        background_percentile=0.0,  # Flow data usually background-subtracted
        norm_quantiles=(0.1, 99.9),
        smoothing_sigma=1.0,
        gradient_method="central_difference"
    )
    
    processed = {}
    for key, field in flow_data.items():
        if field.ndim >= 2:  # Process only spatial data
            result = preprocessor.process_field(field, ca_shape=ca_shape)
            processed[key] = result
        else:
            processed[key] = field
    
    return processed


if __name__ == "__main__":
    # Test field preprocessing
    print("Testing Field Preprocessing...")
    
    # Create synthetic test data
    test_field = np.random.random((20, 64, 64)) * 0.5 + 0.25
    test_field[10:, 20:40, 20:40] += 0.3  # Add some structure
    
    preprocessor = FieldPreprocessor()
    result = preprocessor.process_field(test_field, ca_shape=(32, 32))
    
    print(f"Processed field shape: {result['field'].shape}")
    print(f"Available gradients: {list(result['gradients'].keys())}")
    print(f"Quality metrics: {result['quality_metrics']}")
    
    print("\nField preprocessing test complete!")