"""
Visualization package for second-harmonic generation polarization patterns.

This package provides:
- Consolidated matplotlib configuration
- Reusable 2D polar plotting functions
- 3D surface plotting capabilities
- Logo overlay utilities
- Coordinate conversion and calculation utilities
"""

from .config import configure_matplotlib, is_latex_available
from .plotting import (
    create_2d_polar_plot,
    create_3d_surface_plot,
    add_logo_to_figure,
    plot_polarization_intensity
)
from .utils import (
    spherical_to_cartesian,
    calculate_polarization_intensity,
    sanitize_point_group_name
)

configure_matplotlib()

__all__ = [
    'configure_matplotlib',
    'is_latex_available',
    'create_2d_polar_plot',
    'create_3d_surface_plot',
    'add_logo_to_figure',
    'plot_polarization_intensity',
    'spherical_to_cartesian',
    'calculate_polarization_intensity',
    'sanitize_point_group_name',
]
