#!/usr/bin/env python3
"""
Verification script for visualization pipeline refactoring.
Tests all major refactoring goals.
"""

import sys
import os
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def test_matplotlib_config_consolidated():
    """Test that matplotlib configuration is consolidated."""
    print("Testing matplotlib configuration consolidation...")
    from visualization import configure_matplotlib
    
    # Should work without errors
    configure_matplotlib()
    
    # Check that fonts are configured
    assert 'PingFang SC' in matplotlib.rcParams['font.sans-serif']
    assert matplotlib.rcParams['axes.unicode_minus'] == False
    assert matplotlib.rcParams['font.size'] == 14
    assert matplotlib.rcParams['lines.linewidth'] == 2
    
    print("  ✓ Matplotlib configuration consolidated and working")

def test_reusable_plotting_components():
    """Test that plotting components are reusable."""
    print("Testing reusable plotting components...")
    from visualization import create_2d_polar_plot, create_3d_surface_plot
    from tensor_utils import create_tensor
    
    tensor = create_tensor(['xyz', 'xxy'])
    
    # Test 2D plotting
    fig1, ax1 = create_2d_polar_plot(tensor, "Test Group", phi=0.0, show=False)
    assert fig1 is not None
    assert ax1 is not None
    plt.close(fig1)
    
    # Test 3D plotting
    theta_range = np.linspace(0, np.pi, 20)
    phi_range = np.linspace(0, 2*np.pi, 20)
    fig2, ax2 = create_3d_surface_plot(tensor, theta_range, phi_range, show=False)
    assert fig2 is not None
    assert ax2 is not None
    plt.close(fig2)
    
    print("  ✓ Reusable plotting components working")

def test_error_handling():
    """Test error handling for invalid tensors."""
    print("Testing error handling for invalid tensors...")
    from visualization import create_2d_polar_plot, calculate_polarization_intensity
    
    # Test None tensor
    try:
        create_2d_polar_plot(None, "Test", show=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "不能为空" in str(e)
    
    # Test wrong type
    try:
        calculate_polarization_intensity([1, 2, 3], 0, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "类型不正确" in str(e)
    
    # Test wrong shape
    try:
        create_2d_polar_plot(np.zeros((2, 2)), "Test", show=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "形状不正确" in str(e)
    
    print("  ✓ Error handling working correctly")

def test_backend_independence():
    """Test that figures work without display backend."""
    print("Testing backend independence...")
    from visualization import create_2d_polar_plot
    from tensor_utils import create_tensor
    
    # Should work with Agg backend (no display)
    tensor = create_tensor(['xyz'])
    fig, ax = create_2d_polar_plot(tensor, "Test", show=False)
    assert fig is not None
    
    # Should be able to save to buffer
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png')
    assert buf.tell() > 0  # Some data was written
    
    plt.close(fig)
    print("  ✓ Backend independence confirmed")

def test_deprecated_modules_marked():
    """Test that deprecated modules have warnings."""
    print("Testing deprecated module warnings...")
    import warnings
    
    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from ui import interactive_panel
        assert len(w) > 0
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from ui import plot_widgets
        assert len(w) > 0
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
    
    print("  ✓ Deprecated modules marked with warnings")

def test_backward_compatibility():
    """Test that old imports still work."""
    print("Testing backward compatibility...")
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Old visualization.py import should still work
        from visualization import plot_polarization_intensity
        from tensor_utils import create_tensor
        
        tensor = create_tensor(['xyz'])
        fig = plot_polarization_intensity(tensor, "Test", phi=0.0, show=False)
        assert fig is not None
        plt.close(fig)
    
    print("  ✓ Backward compatibility maintained")

def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")
    from visualization import spherical_to_cartesian, sanitize_point_group_name
    
    # Test coordinate conversion
    x, y, z = spherical_to_cartesian(np.pi/4, np.pi/4)
    assert abs(z - np.sqrt(2)/2) < 1e-5
    
    # Test name sanitization
    name = "C₃ᵥ"
    sanitized = sanitize_point_group_name(name)
    assert "₃" not in sanitized
    assert "ᵥ" not in sanitized
    assert "3" in sanitized
    assert "v" in sanitized
    
    print("  ✓ Utility functions working")

def test_main_app_imports():
    """Test that main app can still import necessary components."""
    print("Testing main app imports...")
    
    # Should be able to import from visualization package
    from visualization import add_logo_to_figure, configure_matplotlib
    
    # Should be able to import other modules
    from point_groups import get_all_point_groups
    from tensor_utils import create_tensor
    from symbolic_calculator import calculate_symbolic_shg_expressions
    
    print("  ✓ Main app imports working")

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Visualization Pipeline Refactoring Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_matplotlib_config_consolidated,
        test_reusable_plotting_components,
        test_error_handling,
        test_backend_independence,
        test_deprecated_modules_marked,
        test_backward_compatibility,
        test_utilities,
        test_main_app_imports,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print()
    print("=" * 60)
    if not failed:
        print("✅ All verification tests passed!")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
