import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.visualization import (
    create_2d_polar_plot,
    create_3d_surface_plot,
    plot_polarization_intensity,
    spherical_to_cartesian,
    calculate_polarization_intensity,
    sanitize_point_group_name
)
from src.tensor_utils import create_tensor

class TestVisualizationUtils(unittest.TestCase):

    def test_spherical_to_cartesian(self):
        theta = np.pi / 4
        phi = np.pi / 4
        x, y, z = spherical_to_cartesian(theta, phi)
        
        self.assertAlmostEqual(x, 0.5, places=5)
        self.assertAlmostEqual(y, 0.5, places=5)
        self.assertAlmostEqual(z, np.sqrt(2) / 2, places=5)
    
    def test_spherical_to_cartesian_array(self):
        theta = np.array([0, np.pi / 2, np.pi])
        phi = np.zeros_like(theta)
        x, y, z = spherical_to_cartesian(theta, phi)
        
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(len(x), 3)
        self.assertAlmostEqual(z[0], 1.0, places=5)
        self.assertAlmostEqual(z[1], 0.0, places=5)
        self.assertAlmostEqual(z[2], -1.0, places=5)
    
    def test_sanitize_point_group_name(self):
        name = "C₃ᵥ"
        sanitized = sanitize_point_group_name(name)
        self.assertEqual(sanitized, "C3v")
        
        name = "D₂ₘ"
        sanitized = sanitize_point_group_name(name)
        self.assertEqual(sanitized, "D2m")
    
    def test_calculate_polarization_intensity_valid(self):
        tensor = create_tensor(['xyz', 'xxy', 'xzx'])
        theta = np.pi / 4
        phi = np.pi / 4
        
        intensity = calculate_polarization_intensity(tensor, theta, phi)
        self.assertIsInstance(intensity, (float, np.floating))
        self.assertGreaterEqual(intensity, 0)
    
    def test_calculate_polarization_intensity_invalid_tensor_none(self):
        with self.assertRaises(ValueError) as context:
            calculate_polarization_intensity(None, 0, 0)
        self.assertIn("不能为空", str(context.exception))
    
    def test_calculate_polarization_intensity_invalid_tensor_type(self):
        with self.assertRaises(ValueError) as context:
            calculate_polarization_intensity([1, 2, 3], 0, 0)
        self.assertIn("类型不正确", str(context.exception))
    
    def test_calculate_polarization_intensity_invalid_tensor_shape(self):
        with self.assertRaises(ValueError) as context:
            calculate_polarization_intensity(np.zeros((2, 2)), 0, 0)
        self.assertIn("形状不正确", str(context.exception))

class TestVisualization2DPlots(unittest.TestCase):

    def setUp(self):
        self.components = ['xyz', 'xxy', 'xzx']
        self.tensor = create_tensor(self.components)
        plt.close('all')

    def tearDown(self):
        plt.close('all')

    def test_create_2d_polar_plot_basic(self):
        fig, ax = create_2d_polar_plot(
            self.tensor, 
            "Test Group", 
            phi=0.0, 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, 'lines'))
        self.assertGreater(len(ax.lines), 0)
    
    def test_create_2d_polar_plot_different_phi(self):
        fig, ax = create_2d_polar_plot(
            self.tensor, 
            "Test Group", 
            phi=45.0, 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIn("45.0", ax.get_title())
    
    def test_create_2d_polar_plot_invalid_tensor_none(self):
        with self.assertRaises(ValueError) as context:
            create_2d_polar_plot(None, "Test", show=False)
        self.assertIn("不能为空", str(context.exception))
    
    def test_create_2d_polar_plot_invalid_tensor_type(self):
        with self.assertRaises(ValueError) as context:
            create_2d_polar_plot([1, 2, 3], "Test", show=False)
        self.assertIn("类型不正确", str(context.exception))
    
    def test_create_2d_polar_plot_invalid_tensor_shape(self):
        with self.assertRaises(ValueError) as context:
            create_2d_polar_plot(np.zeros((2, 2, 2)), "Test", show=False)
        self.assertIn("形状不正确", str(context.exception))
    
    def test_plot_polarization_intensity_legacy(self):
        fig = plot_polarization_intensity(
            self.tensor, 
            "Test Group", 
            phi=0.0, 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'axes'))
    
    def test_create_2d_polar_plot_with_existing_axes(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        fig_returned, ax_returned = create_2d_polar_plot(
            self.tensor,
            "Test Group",
            phi=0.0,
            show=False,
            fig=fig,
            ax=ax
        )
        
        self.assertEqual(fig, fig_returned)
        self.assertEqual(ax, ax_returned)
    
    def test_create_2d_polar_plot_unicode_name(self):
        fig, ax = create_2d_polar_plot(
            self.tensor, 
            "C₃ᵥ", 
            phi=0.0, 
            show=False
        )
        
        title = ax.get_title()
        self.assertIn("C3v", title)
        self.assertNotIn("₃", title)

class TestVisualization3DPlots(unittest.TestCase):

    def setUp(self):
        self.components = ['xyz', 'xxy', 'xzx']
        self.tensor = create_tensor(self.components)
        plt.close('all')

    def tearDown(self):
        plt.close('all')

    def test_create_3d_surface_plot_basic(self):
        fig, ax = create_3d_surface_plot(
            self.tensor, 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, 'zaxis'))
    
    def test_create_3d_surface_plot_custom_ranges(self):
        theta_range = np.linspace(0, np.pi, 20)
        phi_range = np.linspace(0, 2 * np.pi, 20)
        
        fig, ax = create_3d_surface_plot(
            self.tensor,
            theta_range=theta_range,
            phi_range=phi_range,
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_create_3d_surface_plot_invalid_tensor_none(self):
        with self.assertRaises(ValueError) as context:
            create_3d_surface_plot(None, show=False)
        self.assertIn("不能为空", str(context.exception))
    
    def test_create_3d_surface_plot_invalid_tensor_type(self):
        with self.assertRaises(ValueError) as context:
            create_3d_surface_plot([1, 2, 3], show=False)
        self.assertIn("类型不正确", str(context.exception))
    
    def test_create_3d_surface_plot_invalid_tensor_shape(self):
        with self.assertRaises(ValueError) as context:
            create_3d_surface_plot(np.zeros((2, 2)), show=False)
        self.assertIn("形状不正确", str(context.exception))

class TestVisualizationErrorHandling(unittest.TestCase):

    def setUp(self):
        plt.close('all')

    def tearDown(self):
        plt.close('all')

    def test_zero_tensor(self):
        zero_tensor = np.zeros((3, 3, 3))
        fig, ax = create_2d_polar_plot(
            zero_tensor, 
            "Zero Tensor", 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_nan_tensor(self):
        nan_tensor = np.full((3, 3, 3), np.nan)
        
        try:
            fig, ax = create_2d_polar_plot(
                nan_tensor, 
                "NaN Tensor", 
                show=False
            )
            self.assertIsNotNone(fig)
        except Exception:
            pass
    
    def test_large_values_tensor(self):
        large_tensor = np.ones((3, 3, 3)) * 1e10
        fig, ax = create_2d_polar_plot(
            large_tensor, 
            "Large Values", 
            show=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

if __name__ == '__main__':
    unittest.main()
