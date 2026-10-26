"""
绘图画布控件
"""
import numpy as np
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    """Matplotlib绑定的Qt画布"""

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_layout_engine('constrained')
        self.axes = self.fig.add_subplot(111, polar=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.logo_ax_on_fig = None
        self.mode = 'alpha_polar'

    def set_mode(self, mode):
        if mode != self.mode:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111, polar=mode != 'alpha_cartesian')
            self.mode = mode

    def update_plot(self, x_data: np.ndarray, y_data: np.ndarray):
        """更新绑定数据"""
        self.axes.clear()
        x = np.rad2deg(x_data) if self.mode == 'alpha_cartesian' else x_data
        self.axes.plot(x, y_data, lw=2, color='purple')
        if self.mode == 'alpha_cartesian':
            self.axes.set_xlabel('Polarization angle (deg)')
        self.axes.set_ylabel('Relative intensity')
        self.axes.grid(True, linestyle='--', alpha=0.5)
        self.draw()

    def set_title(self, title: str):
        """设置标题"""
        self.axes.set_title(title, y=-0.15, va='top')
        self.draw()
