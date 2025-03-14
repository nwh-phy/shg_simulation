import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class PolarizationPlotWidget:
    def __init__(self, ax, tensor, phi):
        self.ax = ax
        self.tensor = tensor
        self.phi = phi
        self.theta_range = np.linspace(0, 2 * np.pi, 360)
        self.P = self.calculate_polarization()
        self.line, = ax.plot(self.theta_range, self.P, lw=2, color='purple')
        self.ax.set_title(f"Nonlinear Polarization Intensity\nφ={np.rad2deg(self.phi):.1f}°", pad=20)
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # Slider for adjusting tensor size
        self.slider_ax = plt.axes([0.25, 0.01, 0.65, 0.03])
        self.size_slider = Slider(self.slider_ax, 'Tensor Size', 0.1, 5.0, valinit=1.0)
        self.size_slider.on_changed(self.update_plot)

    def calculate_polarization(self):
        P = []
        for theta in self.theta_range:
            E = self.spherical_to_cartesian(theta, self.phi)
            P_i = np.einsum('ijk,j,k->i', self.tensor, E, E)
            P.append(np.linalg.norm(P_i))
        return P

    def spherical_to_cartesian(self, theta, phi):
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))

    def update_plot(self, size):
        scaled_tensor = self.tensor * size
        self.P = self.calculate_polarization()
        self.line.set_ydata(self.P)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()