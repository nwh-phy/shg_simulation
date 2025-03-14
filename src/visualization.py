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
    """
    绘制极化强度极图
    
    参数:
    tensor -- 非线性极化张量
    group_name -- 点群名称
    phi -- 方位角（角度）
    show -- 是否显示图像
    """
    try:
        # 检查输入参数
        if tensor is None:
            print("错误: 张量不能为空")
            return plt.figure()
            
        if not isinstance(tensor, np.ndarray):
            print(f"错误: 张量类型不正确，应为NumPy数组，而非 {type(tensor)}")
            return plt.figure()
            
        if tensor.shape != (3, 3, 3):
            print(f"错误: 张量形状不正确，应为 (3, 3, 3)，而非 {tensor.shape}")
            return plt.figure()
        
        phi_rad = phi * np.pi/180  # 转换为弧度
        theta_range = np.linspace(0, 2*np.pi, 360)
        
        # 计算极化强度
        P = []
        for theta in theta_range:
            try:
                E = spherical_to_cartesian(theta, phi_rad)
                # 张量缩并计算 P_i = χ_ijk E_j E_k 
                P_i = np.einsum('ijk,j,k->i', tensor, E, E)
                P.append(np.linalg.norm(P_i))
            except Exception as e:
                print(f"计算 θ={theta} 的极化强度时出错: {e}")
                P.append(0.0)  # 出错时使用默认值
        
        # 创建新图形
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(111, polar=True)
        
        # 绘制极图
        try:
            ax.plot(theta_range, P, lw=2, color='purple')
        except Exception as e:
            print(f"绘制极图时出错: {e}")
            # 尝试使用备用绘图方法
            ax.scatter(theta_range, P, s=2, color='purple')
        
        # 移除特殊Unicode下标字符，使用普通字符代替
        try:
            display_name = str(group_name)  # 确保是字符串
            display_name = display_name.replace("₁", "1").replace("₂", "2").replace("₃", "3")
            display_name = display_name.replace("₄", "4").replace("₆", "6")
            display_name = display_name.replace("ᵥ", "v").replace("ₕ", "h").replace("ₘ", "m")
            display_name = display_name.replace("ᵤ", "u").replace("ₐ", "a").replace("ᵢ", "i")
            display_name = display_name.replace("̄", "")  # 移除上划线符号
        except Exception as e:
            print(f"处理点群名称时出错: {e}")
            display_name = "Unknown"
        
        # 使用普通文本格式
        title_text = f"{display_name}\n非线性极化强度极图  φ={phi:.1f}°"
            
        ax.set_title(title_text, pad=20)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"显示图形时出错: {e}")
        
        return fig  # 返回图形对象，以便后续操作
    except Exception as e:
        print(f"绘图时发生错误: {e}")
        # 返回一个空的图形对象
        return plt.figure()