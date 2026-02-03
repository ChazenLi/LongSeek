# Archive Directory

## Overview
This directory contains archived scripts, data, and results from earlier stages of the CA Framework development. These files are kept for historical reference and reproducibility but are not part of the active workflow.

## Directory Structure

### `archive/scripts/`
**Archived test and validation scripts** from development iterations.

#### Test Scripts
- `test_simple.py`, `test_simple_v2.py` - Early validation tests
- `test_loading.py` - Data loading functionality tests
- `test_improvements.py`, `test_improvements_final.py`, `test_all_improvements.py` - Iterative improvement tests
- `test_gamma_quick.py`, `test_gamma_full_model.py` - Gamma parameter tests

#### Quick/Example Scripts
- `quick_test.py` - Quick validation test
- `quick_real_data_test.py` - Quick real data test
- `run_example.py` - Basic example script
- `run_synthetic_example.py` - Synthetic data example
- `run_quick_optimization.py` - Quick optimization test

#### Visualization Scripts
- `visualize_real_masks.py` - Mask visualization utility
- `generate_comprehensive_visualization.py` - Old visualization (replaced by `generate_visualization_en.py`)

**Status**: Superseded by modular framework in `examples/` and core workflows

### `archive/data/`
**Archived data and test results**.

#### Test Results
- `results_MDCK_quick.json` - MDCK dataset quick test results
- `results_Melanoma_quick.json` - Melanoma dataset quick test results
- `results_SN15_quick.json` - SN15 dataset quick test results
- `results_TScratch_quick.json` - TScratch dataset quick test results

#### Data Files
- `comparison_dataset.json` - Comparison dataset used in early development
- `observations_reannotation.json` - Reannotated observations

#### Logs
- `output.log` - Early development output logs
- `real_data_output.log` - Real data processing logs

#### Documentation
- `DOCUMENTATION_REPORT.txt` - Early documentation report
- `GAMMA_TEST_RESULTS.txt` - Gamma parameter test results

**Status**: Historical records for reproducibility

### `archive/results_old/`
**Archived visualization results** from earlier iterations.

- `comparison_L12.png` - Early comparison visualization
- `mask_visualization_SN15_*.png` - Mask visualization examples

**Status**: Replaced by newer visualizations in `results/`

## Migration to Active Structure

### Active Scripts
Current workflow scripts are in the root directory:
- `run_optimized_simulation.py` - Main optimized simulation
- `run_grid_search_optimization.py` - Grid search optimization
- `run_bayesian_optimization.py` - Bayesian optimization
- `generate_visualization_en.py` - Current visualization generator
- `run_real_data_analysis.py` - Real data analysis
- `implement_directional_bias.py` - Directional bias implementation
- `multi_dataset_support.py` - Multi-dataset support

### Examples
Validation and example scripts are in `examples/`:
- `minimal_validation.py` - Minimal validation without external data

### Core Framework
The core framework is in `core/`:
- `core/ca/` - CA models
- `core/inference/` - Optimization algorithms
- `core/preprocess/` - Data processing
- `core/utils/` - Utilities

## Recovery Instructions

If you need to recover or reference archived scripts:

1. **Check if functionality has been moved**:
   - Look in `core/` for core functions
   - Look in `examples/` for usage examples
   - Look in `datasets/` for dataset-specific code

2. **If script is truly archived**:
   - Copy from `archive/scripts/` to root
   - Update import paths (old: `from ca.model`, new: `from core.ca`)
   - Test functionality

3. **If data is needed**:
   - Copy from `archive/data/` to appropriate location
   - Update paths in scripts
   - Verify data integrity

## Deletion Policy

Files in `archive/` are kept for:
- **Reproducibility** - Ability to reproduce earlier results
- **Reference** - Understanding development history
- **Recovery** - Potential reuse of code snippets

**Safe to delete** after:
- 6 months of inactivity
- Confirmation that functionality is superseded
- Documentation of key insights in main docs

## Archive Date
Created: 2026-02-02
Last updated: 2026-02-02
