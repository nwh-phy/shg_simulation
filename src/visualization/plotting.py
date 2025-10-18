import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from PyQt5.QtGui import QImage
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False

from .utils import calculate_polarization_intensity, sanitize_point_group_name

def create_2d_polar_plot(tensor, group_name, phi=0.0, show=True, fig=None, ax=None):
    """
    Create a 2D polar plot of polarization intensity.
    
    Parameters:
    -----------
    tensor : ndarray
        3x3x3 nonlinear polarization tensor
    group_name : str
        Point group name for title
    phi : float, optional
        Azimuthal angle in degrees (default: 0.0)
    show : bool, optional
        Whether to display the plot (default: True)
    fig : Figure, optional
        Existing figure to use (default: None, creates new figure)
    ax : Axes, optional
        Existing polar axes to use (default: None, creates new axes)
    
    Returns:
    --------
    tuple
        (fig, ax) Figure and Axes objects
    
    Raises:
    -------
    ValueError
        If tensor is invalid
    """
    if tensor is None:
        raise ValueError("错误: 张量不能为空")
    
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"错误: 张量类型不正确，应为NumPy数组，而非 {type(tensor)}")
    
    if tensor.shape != (3, 3, 3):
        raise ValueError(f"错误: 张量形状不正确，应为 (3, 3, 3)，而非 {tensor.shape}")
    
    phi_rad = phi * np.pi / 180
    theta_range = np.linspace(0, 2 * np.pi, 360)
    
    P = []
    for theta in theta_range:
        try:
            intensity = calculate_polarization_intensity(tensor, theta, phi_rad)
            P.append(intensity)
        except Exception as e:
            print(f"计算 θ={theta} 的极化强度时出错: {e}")
            P.append(0.0)
    
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    
    if ax is None:
        ax = plt.subplot(111, polar=True)
    
    try:
        ax.plot(theta_range, P, lw=2, color='purple')
    except Exception as e:
        print(f"绘制极图时出错: {e}")
        ax.scatter(theta_range, P, s=2, color='purple')
    
    display_name = sanitize_point_group_name(group_name)
    title_text = f"{display_name}\n非线性极化强度极图  φ={phi:.1f}°"
    ax.set_title(title_text, pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图形时出错: {e}")
    
    return fig, ax

def create_3d_surface_plot(tensor, theta_range=None, phi_range=None, show=True):
    """
    Create a 3D surface plot of polarization intensity.
    
    Parameters:
    -----------
    tensor : ndarray
        3x3x3 nonlinear polarization tensor
    theta_range : array-like, optional
        Range of polar angles (default: 0 to 2π)
    phi_range : array-like, optional
        Range of azimuthal angles (default: 0 to 2π)
    show : bool, optional
        Whether to display the plot (default: True)
    
    Returns:
    --------
    tuple
        (fig, ax) Figure and 3D Axes objects
    
    Raises:
    -------
    ValueError
        If tensor is invalid
    """
    if tensor is None:
        raise ValueError("错误: 张量不能为空")
    
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"错误: 张量类型不正确，应为NumPy数组，而非 {type(tensor)}")
    
    if tensor.shape != (3, 3, 3):
        raise ValueError(f"错误: 张量形状不正确，应为 (3, 3, 3)，而非 {tensor.shape}")
    
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 50)
    if phi_range is None:
        phi_range = np.linspace(0, 2 * np.pi, 50)
    
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
    
    R = np.zeros_like(theta_grid)
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            try:
                R[i, j] = calculate_polarization_intensity(tensor, theta_grid[i, j], phi_grid[i, j])
            except Exception as e:
                print(f"计算强度时出错 θ={theta_grid[i,j]}, φ={phi_grid[i,j]}: {e}")
                R[i, j] = 0.0
    
    X = R * np.sin(theta_grid) * np.cos(phi_grid)
    Y = R * np.sin(theta_grid) * np.sin(phi_grid)
    Z = R * np.cos(theta_grid)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5)
    except Exception as e:
        print(f"绘制3D表面时出错: {e}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Polarization Intensity Surface')
    
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图形时出错: {e}")
    
    return fig, ax

def add_logo_to_figure(fig, q_image_logo, position='top-left', size=0.08):
    """
    Add a logo overlay to a matplotlib figure.
    
    Parameters:
    -----------
    fig : Figure
        Matplotlib figure object
    q_image_logo : QImage
        Logo image as PyQt5 QImage
    position : str, optional
        Logo position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        (default: 'top-left')
    size : float, optional
        Relative height of logo as fraction of figure height (default: 0.08)
    
    Returns:
    --------
    Axes or None
        Logo axes object if successful, None otherwise
    """
    if not HAS_PYQT5:
        print("PyQt5 not available, skipping logo drawing.")
        return None
    
    if q_image_logo is None or q_image_logo.isNull():
        print("Logo QImage is null, skipping drawing logo on figure.")
        return None
    
    try:
        ptr = q_image_logo.bits()
        ptr.setsize(q_image_logo.byteCount())
        
        if q_image_logo.format() != QImage.Format_RGBA8888 and q_image_logo.format() != QImage.Format_ARGB32:
            temp_image = q_image_logo.convertToFormat(QImage.Format_RGBA8888)
            if temp_image.isNull():
                print("Error: Could not convert QImage to RGBA8888 for logo.")
                return None
            ptr = temp_image.bits()
            ptr.setsize(temp_image.byteCount())
            logo_numpy_array = np.array(ptr).reshape(temp_image.height(), temp_image.width(), 4)
        else:
            logo_numpy_array = np.array(ptr).reshape(q_image_logo.height(), q_image_logo.width(), 4)
        
        fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
        image_aspect_ratio = q_image_logo.width() / q_image_logo.height()
        
        margin_fig_rel = 0.01
        logo_height_fig_rel = size
        
        logo_width_px = logo_height_fig_rel * fig_height_px * image_aspect_ratio
        logo_width_fig_rel = logo_width_px / fig_width_px
        
        if position == 'top-left':
            rect_left_fig = margin_fig_rel
            rect_bottom_fig = 1.0 - margin_fig_rel - logo_height_fig_rel
            anchor = 'NW'
        elif position == 'top-right':
            rect_left_fig = 1.0 - margin_fig_rel - logo_width_fig_rel
            rect_bottom_fig = 1.0 - margin_fig_rel - logo_height_fig_rel
            anchor = 'NE'
        elif position == 'bottom-left':
            rect_left_fig = margin_fig_rel
            rect_bottom_fig = margin_fig_rel
            anchor = 'SW'
        elif position == 'bottom-right':
            rect_left_fig = 1.0 - margin_fig_rel - logo_width_fig_rel
            rect_bottom_fig = margin_fig_rel
            anchor = 'SE'
        else:
            rect_left_fig = margin_fig_rel
            rect_bottom_fig = 1.0 - margin_fig_rel - logo_height_fig_rel
            anchor = 'NW'
        
        logo_ax = fig.add_axes(
            [rect_left_fig, rect_bottom_fig, logo_width_fig_rel, logo_height_fig_rel],
            anchor=anchor,
            zorder=20
        )
        logo_ax.imshow(logo_numpy_array)
        logo_ax.axis('off')
        
        return logo_ax
        
    except Exception as e:
        print(f"Error drawing logo on figure: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_polarization_intensity(tensor, group_name, phi=0.0, show=True):
    """
    Legacy wrapper function for backward compatibility.
    Plots polarization intensity polar diagram.
    
    Parameters:
    -----------
    tensor : ndarray
        Nonlinear polarization tensor
    group_name : str
        Point group name
    phi : float, optional
        Azimuthal angle in degrees (default: 0.0)
    show : bool, optional
        Whether to display the plot (default: True)
    
    Returns:
    --------
    Figure
        Matplotlib figure object
    """
    try:
        fig, ax = create_2d_polar_plot(tensor, group_name, phi, show)
        return fig
    except Exception as e:
        print(f"绘图时发生错误: {e}")
        return plt.figure()
