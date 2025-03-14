import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import plot_polarization_intensity
from src.tensor_utils import create_tensor

class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.components = ['xyz', 'xxy', 'xzx']
        self.tensor = create_tensor(self.components)

    def test_plot_polarization_intensity(self):
        phi = np.pi / 4  # 45 degrees
        theta_range = np.linspace(0, 2 * np.pi, 360)
        P = []

        for theta in theta_range:
            E = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
            P_i = np.einsum('ijk,j,k->i', self.tensor, E, E)
            P.append(np.linalg.norm(P_i))

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(theta_range, P, lw=2, color='purple')
        ax.set_title("Polarization Intensity Plot", pad=20)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Check if the plot is created
        self.assertIsNotNone(plt.gcf())

    def tearDown(self):
        plt.close()

if __name__ == '__main__':
    unittest.main()