import matplotlib
import shutil

def is_latex_available():
    """Check if LaTeX is available on the system"""
    try:
        return shutil.which('latex') is not None
    except:
        return False

def configure_matplotlib():
    """
    Configure matplotlib with consistent settings for the application.
    Supports Chinese fonts and disables LaTeX to avoid rendering issues.
    """
    matplotlib.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Heiti SC', 'STHeiti',
        'Microsoft YaHei', 'SimHei', 'SimSun',
        'Arial', 'sans-serif'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['lines.linewidth'] = 2
    
    use_latex = False
    return use_latex
