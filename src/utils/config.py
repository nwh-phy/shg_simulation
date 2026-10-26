"""
Matplotlib配置模块
"""
import matplotlib


def configure_matplotlib():
    """配置matplotlib的字体和样式"""
    # 字体设置 - 支持中文显示
    matplotlib.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Heiti SC', 'STHeiti',
        'Microsoft YaHei', 'SimHei', 'SimSun',
        'Arial', 'sans-serif'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['lines.linewidth'] = 2


def is_latex_available() -> bool:
    """检查LaTeX是否可用"""
    try:
        import shutil
        return shutil.which('latex') is not None
    except:
        return False
