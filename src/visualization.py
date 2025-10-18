"""
Legacy visualization module - maintained for backward compatibility.
All functionality has been moved to the visualization package.
"""

import warnings

warnings.warn(
    "The visualization.py module is deprecated. "
    "Please import from 'visualization' package instead.",
    DeprecationWarning,
    stacklevel=2
)

from visualization import (
    configure_matplotlib,
    is_latex_available,
    spherical_to_cartesian,
    plot_polarization_intensity,
    add_logo_to_figure
)

__all__ = [
    'is_latex_available',
    'spherical_to_cartesian',
    'plot_polarization_intensity',
    'add_logo_to_figure'
]
