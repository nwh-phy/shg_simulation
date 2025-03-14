import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class InteractivePanel:
    def __init__(self, tensor, components):
        self.tensor = tensor
        self.components = components
        self.fig, self.ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        self.slider_ax = self.fig.add_axes([0.1, 0.01, 0.8, 0.03])
        self.slider = Slider(self.slider_ax, 'Tensor Scale', 0.1, 5.0, valinit=1.0)

        self.slider.on_changed(self.update_plot)
        self.phi = 0  # Default azimuthal angle
        self.theta_range = np.linspace(0, 2 * np.pi, 360)
        self.plot_initial()

    def plot_initial(self):
        self.P = self.calculate_polarization()
        self.line, = self.ax.plot(self.theta_range, self.P, lw=2, color='purple')
        self.ax.set_title("Nonlinear Polarization Intensity Pole Plot", pad=20)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def calculate_polarization(self):
        P = []
        for theta in self.theta_range:
            E = self.spherical_to_cartesian(theta, self.phi)
            P_i = np.einsum('ijk,j,k->i', self.tensor, E, E)
            P.append(np.linalg.norm(P_i) * self.slider.val)  # Scale by slider value
        return P

    def spherical_to_cartesian(self, theta, phi):
        return (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))

    def update_plot(self, val):
        self.P = self.calculate_polarization()
        self.line.set_ydata(self.P)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

# Example usage (to be integrated into the main application):
# tensor = create_tensor(components)  # Assuming create_tensor is defined elsewhere
# panel = InteractivePanel(tensor, components)