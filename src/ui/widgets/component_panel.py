"""
分量调节面板组件
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal

from ui.widgets.clickable_slider import ClickableSlider


class ComponentPanel(QWidget):
    """分量调节面板"""

    component_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._component_widgets = {}

    def _setup_ui(self):
        """设置UI"""
        self._layout = QVBoxLayout(self)
        self._box = QGroupBox("独立张量参数（关联分量自动更新）")
        self._box_layout = QVBoxLayout(self._box)
        self._layout.addWidget(self._box)

    def update_components(self, components):
        """更新分量列表"""
        # 清除旧控件
        self._clear_widgets()

        # 创建新控件
        for comp in components:
            self._create_slider(comp)

    def _clear_widgets(self):
        """清除所有控件"""
        for widgets in self._component_widgets.values():
            widgets['widget'].deleteLater()
        self._component_widgets.clear()

    def _create_slider(self, comp):
        """创建单个分量滑块"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(f"{comp}:")
        label.setMinimumWidth(40)
        layout.addWidget(label)

        slider = ClickableSlider()
        slider.setMinimum(-1000)
        slider.setMaximum(1000)
        slider.setValue(100)
        slider.setProperty("component", comp)
        slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(slider)

        value_label = QLabel("1.00")
        value_label.setMinimumWidth(40)
        layout.addWidget(value_label)

        self._component_widgets[comp] = {
            'widget': widget,
            'slider': slider,
            'label': value_label
        }
        self._box_layout.addWidget(widget)

    def _on_slider_changed(self):
        """滑块值变化"""
        slider = self.sender()
        comp = slider.property("component")
        value = slider.value() / 100.0

        if comp in self._component_widgets:
            self._component_widgets[comp]['label'].setText(f"{value:.2f}")

        self.component_changed.emit(comp, value)
