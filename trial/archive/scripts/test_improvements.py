"""
Test script for improved CA model with new features:
- Time scaling parameter (k_time)
- Directional migration bias (gamma)
- Enhanced observables
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from CA_project.ca.model import CellOnlyCA, CAParams
from CA_project.preprocess.extract_observations import extract_frame_statistics, calculate_frontline_velocity_series
from CA_project.io_mat import load_binary_mask

def test_enhanced_observables():
    """Test enhanced observables extraction."""
    print("=" * 60)
    print("Testing Enhanced Observables")
    print("=" * 60)
    
    # Create a test mask with a wound
    test_mask = np.ones((100, 100), dtype=np.float32)
    test_mask[30:70, 30:70] = 0.0  # Square wound
    
    # Extract statistics
    stats = extract_frame_statistics(test_mask)
    
    print(f"\nBasic Statistics:")
    print(f"  Cell coverage: {stats['cell_coverage']:.3f}")
    print(f"  Wound area: {stats['wound_area']:.0f}")
    
    print(f"\nRoughness Metrics:")
    print(f"  Perimeter-based: {stats.get('roughness_perimeter', 'N/A'):.3f}")
    print(f"  Height-field: {stats.get('roughness_height_field', 'N/A'):.3f}")
    
    print(f"\nFrontline Width Statistics:")
    print(f"  Mean width: {stats.get('width_mean', 'N/A'):.1f}")
    print(f"  Std width: {stats.get('width_std', 'N/A'):.1f}")
    print(f"  Min/Max width: {stats.get('width_min', 'N/A'):.1f} / {stats.get('width_max', 'N/A'):.1f}")
    
    # Test velocity calculation
    observed_areas = np.array([1600, 1200, 800, 400, 100])
    observed_time = np.array([0, 10, 20, 30, 40])
    velocities = calculate_frontline_velocity_series(observed_areas, observed_time)
    
    print(f"\nVelocity Statistics:")
    print(f"  Mean velocity: {velocities.get('velocity_mean', 'N/A'):.2f}")
    print(f"  Std velocity: {velocities.get('velocity_std', 'N/A'):.2f}")
    for key in velocities:
        if key.startswith('velocity_'):
            print(f"  {key}: {velocities[key]:.2f}")

def test_time_scaling():
    """Test time scaling parameter k_time."""
    print("\n" + "=" * 60)
    print("Testing Time Scaling (k_time)")
    print("=" * 60)
    
    # Create initial mask
    initial_mask = np.ones((128, 128), dtype=np.float32)
    initial_mask[40:88, 40:88] = 0.0  # Square wound
    
    # Test with different k_time values
    observed_time = np.array([0, 17, 40])
    
    for k_time in [0.5, 1.0, 2.0]:
        params = CAParams(p_move=0.5, p_div=0.05, k_time=k_time)
        ca = CellOnlyCA(128, 128, params)
        ca.initialize_from_mask(initial_mask, k=1)
        
        # Calculate number of CA steps needed
        num_steps = int(observed_time[-1] * k_time) + 10
        
        print(f"\nk_time = {k_time}")
        print(f"  Real time range: 0 to {observed_time[-1]}")
        print(f"  CA steps needed: {num_steps}")
        print(f"  Time scaling: 1 CA step = {1/k_time:.2f} real time units")

def test_directional_migration():
    """Test directional migration bias (gamma)."""
    print("\n" + "=" * 60)
    print("Testing Directional Migration Bias (gamma)")
    print("=" * 60)
    
    # Create initial mask with a wound
    initial_mask = np.ones((64, 64), dtype=np.float32)
    initial_mask[20:44, 20:44] = 0.0  # Square wound in center
    
    # Test with different gamma values
    for gamma in [0.0, 0.5, 1.0]:
        params = CAParams(p_move=0.8, p_div=0.05, gamma=gamma)
        ca = CellOnlyCA(64, 64, params)
        ca.initialize_from_mask(initial_mask, k=1)
        
        # Run simulation for a few steps
        history = ca.run(num_steps=20)
        
        initial_area = history[0]['wound_area']
        final_area = history[-1]['wound_area']
        closure_rate = (initial_area - final_area) / initial_area * 100
        
        print(f"\ngamma = {gamma}")
        print(f"  Initial wound area: {initial_area:.0f}")
        print(f"  Final wound area: {final_area:.0f}")
        print(f"  Closure rate: {closure_rate:.1f}%")
        print(f"  Total migrations: {sum(s['migrations'] for s in history)}")
        print(f"  Total divisions: {sum(s['divisions'] for s in history)}")

def test_combined_improvements():
    """Test all improvements together."""
    print("\n" + "=" * 60)
    print("Testing Combined Improvements")
    print("=" * 60)
    
    # Create initial mask
    initial_mask = np.ones((128, 128), dtype=np.float32)
    initial_mask[40:88, 40:88] = 0.0  # Square wound
    
    # Parameters with improvements enabled
    params = CAParams(
        p_move=0.7,
        p_div=0.08,
        alpha=1.0,
        beta=1.0,
        edge_bonus=2.0,
        k_time=1.5,  # Time scaling: CA runs 1.5x faster than real time
        gamma=0.8    # Strong directional bias toward wound
    )
    
    print(f"\nParameters:")
    print(f"  p_move: {params.p_move}")
    print(f"  p_div: {params.p_div}")
    print(f"  k_time: {params.k_time}")
    print(f"  gamma: {params.gamma}")
    
    # Create and run CA
    ca = CellOnlyCA(128, 128, params)
    ca.initialize_from_mask(initial_mask, k=1)
    
    # Run simulation
    num_ca_steps = int(40 * params.k_time) + 10  # Simulate t=0 to t=40
    history = ca.run(num_steps=num_ca_steps)
    
    print(f"\nSimulation Results:")
    print(f"  CA steps: {len(history)}")
    print(f"  Real time equivalent: {len(history) / params.k_time:.1f}")
    print(f"  Initial wound area: {history[0]['wound_area']:.0f}")
    print(f"  Final wound area: {history[-1]['wound_area']:.0f}")
    print(f"  Closure percentage: {(1 - history[-1]['wound_area']/history[0]['wound_area']) * 100:.1f}%")
    
    # Compare at observed time points
    observed_time = np.array([0, 17, 40])
    sim_time_points = (observed_time * params.k_time).astype(int)
    
    print(f"\nComparison at Observed Time Points:")
    for t_real, t_sim in zip(observed_time, sim_time_points):
        if t_sim < len(history):
            area = history[t_sim]['wound_area']
            normalized = area / history[0]['wound_area']
            print(f"  t={t_real:2d} (CA step {t_sim:2d}): A(t)={area:6.0f} ({normalized:.3f})")

def test_with_real_data():
    """Test with real SN15 data if available."""
    print("\n" + "=" * 60)
    print("Testing with Real SN15 Data")
    print("=" * 60)
    
    try:
        # Try to load L12 data
        mask_path = "CA/DATA/SN15/SN15/manual_mat/DIC_SN_15_L12_Sum00_manual.mat"
        mask = load_binary_mask(mask_path)
        
        print(f"\nLoaded mask: {mask_path}")
        print(f"  Shape: {mask.shape}")
        print(f"  Data type: {mask.dtype}")
        
        # Extract observables
        stats = extract_frame_statistics(mask)
        print(f"\nObservables:")
        print(f"  Cell coverage: {stats['cell_coverage']:.3f}")
        print(f"  Wound area: {stats['wound_area']:.0f}")
        print(f"  Width statistics: mean={stats['width_mean']:.1f}, std={stats['width_std']:.1f}")
        print(f"  Roughness (height field): {stats['roughness_height_field']:.3f}")
        
        # Run short simulation
        params = CAParams(
            p_move=0.7, p_div=0.08,
            k_time=1.5, gamma=0.8
        )
        
        ca = CellOnlyCA(128, 128, params)
        ca.initialize_from_mask(mask, k=8)  # Downsample 1024 -> 128
        
        history = ca.run(num_steps=60)
        
        print(f"\nShort Simulation (60 CA steps):")
        print(f"  Initial area: {history[0]['wound_area']:.0f}")
        print(f"  Final area: {history[-1]['wound_area']:.0f}")
        print(f"  Closure: {(1 - history[-1]['wound_area']/history[0]['wound_area']) * 100:.1f}%")
        
    except Exception as e:
        print(f"\nCould not load real data: {e}")
        print("This is expected if data paths are not set up yet.")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CA Model Improvements Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_enhanced_observables()
    test_time_scaling()
    test_directional_migration()
    test_combined_improvements()
    test_with_real_data()
    
    print("\n" + "=" * 60)
    print("All Tests Complete!")
    print("=" * 60)
