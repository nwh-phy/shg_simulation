import numpy as np
import matplotlib.pyplot as plt
from src.visualization import plot_polarization_intensity


def test_legacy_plot_obeys_quadratic_tensor_scaling_and_transverse_detection():
    tensor = np.zeros((3, 3, 3))
    tensor[0, 0, 0] = 1
    tensor[2, 0, 0] = 100
    first = plot_polarization_intensity(tensor, 'synthetic tensor', show=False)
    second = plot_polarization_intensity(2*tensor, 'synthetic tensor', show=False)
    a, b = first.axes[0].lines[0].get_ydata(), second.axes[0].lines[0].get_ydata()
    assert a[0] == 1
    np.testing.assert_allclose(b, 4*a)
    plt.close(first); plt.close(second)
