import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 检查LaTeX是否可用 - 更安全的方法
def is_latex_available():
    try:
        # 简单检查是否安装了latex命令
        import shutil
        return shutil.which('latex') is not None
    except:
        return False

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']  # 优先使用更完整的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2

# 禁用LaTeX，使用普通文本渲染
use_latex = False  # 直接设置为False，避免LaTeX相关问题

def spherical_to_cartesian(theta, phi):
    """球坐标转笛卡尔坐标"""
    try:
        return (np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta))
    except Exception as e:
        print(f"坐标转换错误: {e}")
        # 返回一个默认值
        return (0, 0, 1)

def plot_polarization_intensity(tensor, group_name, phi=0.0, show=True):
    """Legacy plot helper: incident-angle scan, transverse intensity in arbitrary units."""
    try:
        from .core.shg_calculator import SHGCalculator, ScanParameters
        from .core.tensor_operations import TensorOperations
    except ImportError:
        from core.shg_calculator import SHGCalculator, ScanParameters
        from core.tensor_operations import TensorOperations
    tensor = np.asarray(tensor)
    if tensor.shape != (3, 3, 3) or not np.isfinite(tensor).all():
        raise ValueError('A finite 3x3x3 tensor is required.')
    theta = np.linspace(0, 2*np.pi, 361)
    intensity = SHGCalculator(TensorOperations.tensor_to_voigt(tensor)).calculate_intensity_vectorized(ScanParameters(theta, np.deg2rad(phi)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta, intensity, color='purple')
    ax.set_title(f'{group_name}: incident-angle scan, phi={phi:g} deg')
    ax.set_ylabel('Relative transverse intensity')
    ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return fig
