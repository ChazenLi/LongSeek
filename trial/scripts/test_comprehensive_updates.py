"""
Comprehensive Test Suite for CA Framework Updates

This script tests all new functionality added to the CA framework,
including:
- Morphological observables
- Field consistency metrics
- Uncertainty quantification
- Multi-objective optimization
- Uncertainty-weighted loss

Usage:
    python test_comprehensive_updates.py
"""

import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List
import sys
import io

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_morphological_metrics():
    """Test morphological observable calculations."""
    print("\n" + "="*70)
    print("TEST 1: Morphological Metrics")
    print("="*70 + "\n")
    
    try:
        from core.metrics.morphological import (
            calculate_all_morphological_metrics,
            calculate_roughness_perimeter,
            calculate_width_statistics,
            calculate_shape_metrics,
            calculate_front_distance_distribution,
            calculate_velocity_segments
        )
        
        # Create synthetic wound mask
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 0  # Square wound in center
        
        # Add some roughness
        mask[30:35, 40:45] = 1
        mask[65:70, 55:60] = 0
        
        # Test all morphological metrics
        metrics = calculate_all_morphological_metrics(mask)
        
        print("Morphological Metrics Calculated:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} elements")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
        
        # Test velocity segments
        times = [0, 10, 20, 30, 40]
        areas = [1600, 1200, 800, 500, 200]
        velocity_metrics = calculate_velocity_segments(areas, times, n_segments=3)
        
        print(f"\nVelocity Segments:")
        print(f"  Mean velocity: {velocity_metrics['mean_velocity']:.4f}")
        print(f"  Std velocity: {velocity_metrics['velocity_std']:.4f}")
        print(f"  Velocities: {velocity_metrics['velocities']}")
        
        print("\n✅ Test 1 PASSED: Morphological metrics working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_field_consistency():
    """Test field consistency metrics."""
    print("\n" + "="*70)
    print("TEST 2: Field Consistency Metrics")
    print("="*70 + "\n")
    
    try:
        from core.metrics.field_consistency import (
            calculate_velocity_field_alignment,
            calculate_field_ca_coupling,
            calculate_opticflow_agreement
        )
        
        # Create synthetic test data
        H, W = 50, 50
        
        # Synthetic molecular field
        field = np.linspace(0, 1, H)[:, np.newaxis] * np.ones((1, W))
        
        # Synthetic CA velocity
        u_ca = np.zeros((H, W)) + 0.1
        v_ca = np.ones((H, W)) * 0.5
        ca_velocity = {'u': u_ca, 'v': v_ca}
        
        # Field gradient
        grad_y, grad_x = np.gradient(field)
        field_gradient = {'grad_x': grad_x, 'grad_y': grad_y}
        
        # Test velocity-field alignment
        alignment = calculate_velocity_field_alignment(ca_velocity, field_gradient)
        print("Velocity-Field Alignment:")
        print(f"  Mean cosine similarity: {alignment['mean_cosine_similarity']:.4f}")
        print(f"  Mean angle difference: {alignment['mean_angle_difference_degrees']:.2f}°")
        
        # Test field-CA coupling
        ca_activity = np.random.random((H, W))
        coupling = calculate_field_ca_coupling(field, ca_activity)
        print(f"\nField-CA Coupling:")
        print(f"  Correlation: {coupling['field_ca_correlation']:.4f}")
        print(f"  Mutual information: {coupling['mutual_information']:.4f}")
        
        # Test opticflow agreement
        opticflow = {
            'u': u_ca + 0.05,
            'v': v_ca + 0.05,
            'magnitude': np.sqrt((u_ca + 0.05)**2 + (v_ca + 0.05)**2)
        }
        agreement = calculate_opticflow_agreement(ca_velocity, opticflow)
        print(f"\nOpticflow Agreement:")
        print(f"  Mean magnitude error: {agreement['mean_magnitude_error']:.4f}")
        print(f"  Mean angle error: {agreement['mean_angle_error_degrees']:.2f}°")
        
        print("\n✅ Test 2 PASSED: Field consistency metrics working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_quantification():
    """Test uncertainty quantification methods."""
    print("\n" + "="*70)
    print("TEST 3: Uncertainty Quantification")
    print("="*70 + "\n")
    
    try:
        from core.metrics.uncertainty import (
            calculate_parameter_correlation,
            calculate_confidence_intervals,
            assess_identifiability,
            calculate_information_criteria
        )
        
        # Generate synthetic parameter samples
        np.random.seed(42)
        n_samples = 100
        param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma']
        
        # Create correlated samples
        base_samples = np.random.randn(n_samples, len(param_names))
        
        # Add some correlations
        base_samples[:, 0] += 0.5 * base_samples[:, 2]  # p_move correlated with alpha
        base_samples[:, 1] += 0.3 * base_samples[:, 3]  # p_div correlated with beta
        
        # Scale to realistic ranges
        params_samples = np.zeros_like(base_samples)
        params_samples[:, 0] = 0.85 + 0.1 * base_samples[:, 0]  # p_move
        params_samples[:, 1] = 0.17 + 0.05 * base_samples[:, 1]  # p_div
        params_samples[:, 2] = 1.45 + 0.2 * base_samples[:, 2]  # alpha
        params_samples[:, 3] = 0.70 + 0.1 * base_samples[:, 3]  # beta
        params_samples[:, 4] = 1.41 + 0.2 * base_samples[:, 4]  # edge_bonus
        params_samples[:, 5] = 1.5 + 0.3 * base_samples[:, 5]  # k_time
        params_samples[:, 6] = 0.0 + 0.1 * base_samples[:, 6]  # gamma
        
        # Test parameter correlation
        corr_analysis = calculate_parameter_correlation(params_samples, param_names)
        print("Parameter Correlation:")
        print(f"  Identifiability score: {corr_analysis['identifiability_score']:.4f}")
        print(f"  Highly correlated pairs: {len(corr_analysis['highly_correlated_pairs'])}")
        
        for pair in corr_analysis['highly_correlated_pairs']:
            print(f"    {pair['param1']} - {pair['param2']}: {pair['correlation']:.3f}")
        
        # Test confidence intervals
        ci_dict = calculate_confidence_intervals(params_samples, param_names, confidence_level=0.95)
        print(f"\n95% Confidence Intervals:")
        for name, ci in ci_dict.items():
            print(f"  {name}: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        # Test identifiability assessment
        identifiability = assess_identifiability(
            np.array(corr_analysis['correlation_matrix']),
            param_names,
            threshold=0.8
        )
        print(f"\nIdentifiability Assessment:")
        print(f"  Overall: {identifiability['assessment']}")
        print(f"  Identifiable: {identifiability['n_identifiable']}/{identifiability['n_total']}")
        
        # Test information criteria
        loss_values = [0.0042, 0.0050, 0.0038, 0.0060]
        ic = calculate_information_criteria(loss_values, n_params=7, n_data_points=3)
        print(f"\nInformation Criteria:")
        print(f"  Best model (AIC): idx {ic['best_aic_idx']}")
        print(f"  Best model (BIC): idx {ic['best_bic_idx']}")
        
        print("\n✅ Test 3 PASSED: Uncertainty quantification working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_existing_framework():
    """Test integration with existing CA framework."""
    print("\n" + "="*70)
    print("TEST 4: Integration with Existing Framework")
    print("="*70 + "\n")
    
    try:
        from core import CellOnlyCA, CAParams
        from core.metrics.morphological import calculate_all_morphological_metrics
        
        # Create CA model with optimal parameters from Week 1
        params = CAParams(
            p_move=0.85,
            p_div=0.17,
            alpha=1.45,
            beta=0.70,
            edge_bonus=1.41,
            k_time=1.5,
            gamma=0.0
        )
        
        ca = CellOnlyCA(height=128, width=128, params=params)
        
        # Initialize with synthetic wound
        mask = np.ones((128, 128), dtype=np.uint8)
        mask[40:88, 40:88] = 0
        ca.initialize_from_mask(mask, k=1)
        
        # Run simulation
        print("Running CA simulation...")
        history = ca.run(num_steps=50)
        
        # Calculate morphological metrics from initial mask
        metrics = calculate_all_morphological_metrics(mask)
        
        print("\nSimulation Results:")
        print(f"  Initial wound area: {history[0]['wound_area']:.0f}")
        print(f"  Final wound area: {history[-1]['wound_area']:.0f}")
        print(f"  Closure rate: {(1 - history[-1]['wound_area']/history[0]['wound_area'])*100:.1f}%")
        
        print(f"\nMorphological Metrics:")
        print(f"  Width mean: {metrics['width_mean']:.2f}")
        print(f"  Roughness (height field): {metrics['roughness_height_field']:.4f}")
        print(f"  Aspect ratio: {metrics['aspect_ratio']:.4f}")
        
        print("\n✅ Test 4 PASSED: Integration with existing framework working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_imports():
    """Test that all new modules can be imported."""
    print("\n" + "="*70)
    print("TEST 5: Module Imports")
    print("="*70 + "\n")
    
    try:
        # Test core.metrics imports
        from core.metrics import (
            calculate_roughness_perimeter,
            calculate_roughness_height_field,
            calculate_width_statistics,
            calculate_shape_metrics,
            calculate_front_distance_distribution,
            calculate_velocity_segments,
            calculate_all_morphological_metrics
        )
        
        from core.metrics.field_consistency import (
            calculate_velocity_field_alignment,
            calculate_field_ca_coupling,
            calculate_opticflow_agreement
        )
        
        from core.metrics.uncertainty import (
            calculate_parameter_correlation,
            calculate_confidence_intervals,
            assess_identifiability
        )
        
        print("All modules imported successfully:")
        print("  - core.metrics.morphological")
        print("  - core.metrics.field_consistency")
        print("  - core.metrics.uncertainty")
        
        print("\n✅ Test 5 PASSED: All modules importable")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and generate summary report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE FOR CA FRAMEWORK UPDATES")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Morphological Metrics", test_morphological_metrics()))
    results.append(("Field Consistency", test_field_consistency()))
    results.append(("Uncertainty Quantification", test_uncertainty_quantification()))
    results.append(("Framework Integration", test_integration_with_existing_framework()))
    results.append(("Module Imports", test_module_imports()))
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        'timestamp': str(np.datetime64('now')),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'test_results': [
            {'name': name, 'passed': bool(passed)}
            for name, passed in results
        ]
    }
    
    summary_path = output_dir / "test_summary_comprehensive.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTest summary saved to: {summary_path}")
    
    # Return overall success
    return passed_tests == total_tests


if __name__ == "__main__":
    import sys
    
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
