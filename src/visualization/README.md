# Visualization Package

Unified visualization module for second-harmonic generation polarization patterns.

## Structure

- `config.py` - Consolidated matplotlib configuration
- `utils.py` - Coordinate conversion and calculation utilities
- `plotting.py` - 2D polar plots, 3D surface plots, and logo overlay functions
- `__init__.py` - Package initialization and public API

## Features

### Matplotlib Configuration
Centralized configuration for consistent rendering across the application:
- Chinese font support (macOS and Windows)
- Unicode minus sign handling
- Standard font sizes and line widths
- LaTeX availability checking

### Coordinate Utilities
- `spherical_to_cartesian(theta, phi)` - Convert spherical to Cartesian coordinates
- `calculate_polarization_intensity(tensor, theta, phi)` - Calculate SHG polarization intensity
- `sanitize_point_group_name(name)` - Remove Unicode subscripts/superscripts for display

### 2D Polar Plots
- `create_2d_polar_plot(tensor, group_name, phi, show, fig, ax)` - Create interactive polar plots
- Supports custom or existing figure/axes
- Error handling for invalid tensors
- Automatic point group name sanitization

### 3D Surface Plots
- `create_3d_surface_plot(tensor, theta_range, phi_range, show)` - Create 3D intensity surfaces
- Customizable angle ranges
- Colormap support
- Error handling for invalid inputs

### Logo Overlay
- `add_logo_to_figure(fig, q_image_logo, position, size)` - Add logos to matplotlib figures
- Multiple position options (top-left, top-right, bottom-left, bottom-right)
- Configurable size
- PyQt5 QImage support

## Usage

```python
from visualization import (
    configure_matplotlib,
    create_2d_polar_plot,
    create_3d_surface_plot,
    add_logo_to_figure
)

# Configure matplotlib (done automatically on import)
configure_matplotlib()

# Create a 2D polar plot
fig, ax = create_2d_polar_plot(
    tensor=my_tensor,
    group_name="C3v",
    phi=45.0,
    show=False
)

# Create a 3D surface plot
fig_3d, ax_3d = create_3d_surface_plot(
    tensor=my_tensor,
    show=False
)

# Add logo to figure (requires PyQt5)
logo_ax = add_logo_to_figure(
    fig=fig,
    q_image_logo=my_qimage,
    position='top-left',
    size=0.08
)
```

## Error Handling

All plotting functions validate input tensors and raise `ValueError` with descriptive Chinese messages for:
- None tensors
- Invalid tensor types (non-ndarray)
- Invalid tensor shapes (not 3×3×3)

Functions also handle edge cases gracefully:
- Zero tensors
- NaN values
- Extremely large values

## Testing

Comprehensive test suite in `tests/test_visualization.py` covers:
- Coordinate conversion utilities
- 2D polar plot creation and error handling
- 3D surface plot creation and error handling
- Edge cases (zero, NaN, large values)
- Unicode name sanitization
- Backend-independent rendering (uses Agg backend)

Run tests with:
```bash
python -m unittest tests.test_visualization -v
```

## Migration Guide

### From `src/visualization.py`
The old `visualization.py` module is deprecated. Update imports:

```python
# Old
from visualization import plot_polarization_intensity

# New
from visualization import plot_polarization_intensity  # Still works via wrapper
# OR
from visualization import create_2d_polar_plot  # New function with more features
```

### From `src/ui/interactive_panel.py` and `src/ui/plot_widgets.py`
These modules are deprecated. Replace with:

```python
# Old
from ui.interactive_panel import InteractivePanel

# New
from visualization import create_2d_polar_plot
fig, ax = create_2d_polar_plot(tensor, "Group Name", phi=0.0, show=True)
```

## Dependencies

- numpy
- matplotlib
- PyQt5 (optional, for logo overlay functionality)
