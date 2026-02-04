"""
Metrics and Observables for CA Model Evaluation

This module provides comprehensive metrics for evaluating CA simulations,
including morphological observables, field consistency metrics, and
uncertainty quantification.

Core Components:
- morphological.py: Morphological observables (roughness, width, shape metrics)
- field_consistency.py: Field consistency metrics for molecular fields
- uncertainty.py: Uncertainty quantification and confidence intervals
"""

from .morphological import (
    calculate_roughness_perimeter,
    calculate_roughness_height_field,
    calculate_width_statistics,
    calculate_shape_metrics,
    calculate_front_distance_distribution,
    calculate_velocity_segments,
    calculate_all_morphological_metrics
)

__all__ = [
    'calculate_roughness_perimeter',
    'calculate_roughness_height_field',
    'calculate_width_statistics',
    'calculate_shape_metrics',
    'calculate_front_distance_distribution',
    'calculate_velocity_segments',
    'calculate_all_morphological_metrics',
]
