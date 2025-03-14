import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.point_groups import point_group_components
from src.tensor_utils import create_tensor, str_to_indices
from src.visualization import plot_polarization_intensity

def interactive_tensor_adjustment():
    # 设置初始参数
    initial_group = '1 (triclinic)'
    components = point_group_components[initial_group]['Non-zero components']
    tensor = create_tensor(components)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # 绘制初始极化强度极图
    phi = np.pi / 4  # 固定方位角
    plot_polarization_intensity(ax, tensor, phi)

    # 添加滑块以调整张量大小
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Tensor Scale', 0.1, 5.0, valinit=1.0)

    def update(val):
        scale = slider.val
        scaled_tensor = tensor * scale
        ax.clear()
        plot_polarization_intensity(ax, scaled_tensor, phi)
        ax.set_title(f"Scaled Tensor Polarization Intensity (Scale: {scale:.2f})")
        plt.draw()

    slider.on_changed(update)

    plt.title(f"{initial_group} Nonlinear Polarization Intensity")
    plt.show()

if __name__ == "__main__":
    interactive_tensor_adjustment()