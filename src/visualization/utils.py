import numpy as np

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Parameters:
    -----------
    theta : float or array
        Polar angle in radians
    phi : float or array
        Azimuthal angle in radians
    
    Returns:
    --------
    tuple
        (x, y, z) Cartesian coordinates
    
    Raises:
    -------
    ValueError
        If coordinate conversion fails
    """
    try:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return (x, y, z)
    except Exception as e:
        raise ValueError(f"坐标转换错误: {e}")

def calculate_polarization_intensity(tensor, theta, phi):
    """
    Calculate polarization intensity for given angles.
    
    Parameters:
    -----------
    tensor : ndarray
        3x3x3 nonlinear polarization tensor
    theta : float or array
        Polar angle in radians
    phi : float or array
        Azimuthal angle in radians
    
    Returns:
    --------
    float or array
        Polarization intensity magnitude
    
    Raises:
    -------
    ValueError
        If tensor is invalid or calculation fails
    """
    if tensor is None:
        raise ValueError("错误: 张量不能为空")
    
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"错误: 张量类型不正确，应为NumPy数组，而非 {type(tensor)}")
    
    if tensor.shape != (3, 3, 3):
        raise ValueError(f"错误: 张量形状不正确，应为 (3, 3, 3)，而非 {tensor.shape}")
    
    try:
        E = spherical_to_cartesian(theta, phi)
        P_i = np.einsum('ijk,j,k->i', tensor, E, E)
        return np.linalg.norm(P_i)
    except Exception as e:
        raise ValueError(f"计算极化强度时出错: {e}")

def sanitize_point_group_name(group_name):
    """
    Remove special Unicode characters from point group names for display.
    
    Parameters:
    -----------
    group_name : str
        Point group name with Unicode subscripts and superscripts
    
    Returns:
    --------
    str
        Sanitized name with ASCII characters
    """
    try:
        display_name = str(group_name)
        display_name = display_name.replace("₁", "1").replace("₂", "2").replace("₃", "3")
        display_name = display_name.replace("₄", "4").replace("₆", "6")
        display_name = display_name.replace("ᵥ", "v").replace("ₕ", "h").replace("ₘ", "m")
        display_name = display_name.replace("ᵤ", "u").replace("ₐ", "a").replace("ᵢ", "i")
        display_name = display_name.replace("̄", "")
        return display_name
    except Exception as e:
        print(f"处理点群名称时出错: {e}")
        return "Unknown"
