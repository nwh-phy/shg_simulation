import numpy as np
import matplotlib.pyplot as plt
from src.point_groups import point_group_components
from src.tensor_utils import create_tensor, spherical_to_cartesian
from src.ui.interactive_panel import InteractivePanel

def basic_simulation():
    print("可用晶体点群:")
    groups = list(point_group_components.keys())
    for i, g in enumerate(groups, 1):
        print(f"{i}. {g}")
    
    group_choice = int(input("请选择点群编号: ")) - 1
    selected_group = groups[group_choice]
    
    components = point_group_components[selected_group]['Non-zero components']
    tensor = create_tensor(components)
    
    phi = float(input("输入方位角φ (度): ")) * np.pi / 180
    theta_range = np.linspace(0, 2 * np.pi, 360)
    
    P = []
    for theta in theta_range:
        E = spherical_to_cartesian(theta, phi)
        P_i = np.einsum('ijk,j,k->i', tensor, E, E)
        P.append(np.linalg.norm(P_i))
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta_range, P, lw=2, color='purple')
    ax.set_title(f"{selected_group} NonlinearPolarizationIntensityPolePlot\nφ={np.rad2deg(phi):.1f}°", pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    panel = InteractivePanel(basic_simulation)
    panel.run()