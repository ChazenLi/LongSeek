"""
CA Framework - Core Module
A modular cellular automata framework for wound healing modeling.
"""

__version__ = "3.0.0"
__author__ = "CA Project Team"

from .ca import CellOnlyCA, CAParams, CAMolParams, CAMolField
from .inference import run_simulation, calculate_loss, fit_simple_grid_search, fit_bayesian_optimization
from .preprocess import extract_observations, downsample_binary, extract_frame_statistics
from .utils import load_binary_mask

# Visualization module (Week 9)
try:
    from .visualization import (
        setup_paper_style,
        IFCAColors,
        plot_parameter_search,
        plot_ablation_study,
        plot_identifiability_diagnostics,
        plot_multi_condition_training,
        plot_causal_validation,
        plot_summary_dashboard,
    )
    _has_viz = True
except ImportError:
    _has_viz = False

__all__ = [
    # CA Models
    'CellOnlyCA',
    'CAParams',
    'CAMolParams',
    'CAMolField',

    # Inference
    'run_simulation',
    'calculate_loss',
    'fit_simple_grid_search',
    'fit_bayesian_optimization',

    # Preprocessing
    'extract_observations',
    'downsample_binary',
    'extract_frame_statistics',

    # Utilities
    'load_binary_mask',

    # Visualization (conditional)
    'setup_paper_style',
    'IFCAColors',
    'plot_parameter_search',
    'plot_ablation_study',
    'plot_identifiability_diagnostics',
    'plot_multi_condition_training',
    'plot_causal_validation',
    'plot_summary_dashboard',
]
