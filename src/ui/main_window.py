"""
新主窗口 - 使用MVP架构
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QScrollArea, QMessageBox
)

from ui.widgets.plot_canvas import PlotCanvas
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.component_panel import ComponentPanel
from ui.controllers.main_controller import MainController


class NewMainWindow(QMainWindow):
    """新主窗口"""

    def __init__(self):
        super().__init__()
        self._controller = MainController()
        self._setup_ui()
        self._connect_signals()
        self._initialize()

    def _setup_ui(self):
        """设置UI"""
        self.setWindowTitle("SHG 张量模拟")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        self._create_control_panel(main_layout)

        # 右侧绑定区域
        self._canvas = PlotCanvas(width=8, height=8)
        main_layout.addWidget(self._canvas, 2)

    def _create_control_panel(self, parent_layout):
        """创建控制面板"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)

        self._param_panel = ParameterPanel()
        layout.addWidget(self._param_panel)

        self._comp_panel = ComponentPanel()
        layout.addWidget(self._comp_panel)

        # 按钮
        self._reset_btn = QPushButton("重置参数")
        layout.addWidget(self._reset_btn)
        self._import_experiment_btn = QPushButton("导入实验数据 / 提取 2ω 强度")
        self._import_experiment_btn.clicked.connect(self._open_experimental_import)
        layout.addWidget(self._import_experiment_btn)
        symbolic = QPushButton("查看当前取向的解析式")
        symbolic.clicked.connect(self._show_symbolic)
        layout.addWidget(symbolic)

        layout.addStretch()
        scroll.setWidget(container)
        parent_layout.addWidget(scroll, 1)

    def _connect_signals(self):
        """连接信号"""
        # 控制器 -> UI
        self._controller.plot_data_ready.connect(self._on_plot_data)
        self._controller.components_updated.connect(self._comp_panel.update_components)
        self._controller.error_occurred.connect(self._show_error)
        self._controller.state_changed.connect(self._sync_state)

        # UI -> 控制器
        self._param_panel.point_group_changed.connect(self._controller.set_point_group)
        self._param_panel.orientation_changed.connect(self._controller.set_orientation)
        self._param_panel.global_scale_changed.connect(self._controller.set_global_scale)
        self._param_panel.scan_mode_changed.connect(lambda value: self._controller.set_parameter('scan_mode', value))
        self._param_panel.phi_beam_changed.connect(lambda value: self._controller.set_parameter('phi_beam', value))
        self._param_panel.parameter_changed.connect(self._controller.set_parameter)
        self._comp_panel.component_changed.connect(self._controller.set_component_value)
        self._reset_btn.clicked.connect(self._controller.reset_parameters)

    def _initialize(self):
        """初始化"""
        from services.point_group_service import PointGroupService
        service = PointGroupService()
        groups = service.get_all_groups()
        self._param_panel.set_point_groups(groups)

    def _open_experimental_import(self):
        from ui.experimental_import import show_experimental_import
        show_experimental_import(self)

    def _on_plot_data(self, x, y):
        mode = self._controller.state.scan_mode
        self._canvas.set_mode(mode)
        self._canvas.update_plot(x, y)

    def _sync_state(self):
        from PyQt5.QtCore import QSignalBlocker
        state = self._controller.state
        self._param_panel.sync_state(state)
        for comp, widgets in self._comp_panel._component_widgets.items():
            blocker = QSignalBlocker(widgets['slider'])
            widgets['slider'].setValue(int(round(state.component_values[comp]*100)))
            widgets['label'].setText(f'{state.component_values[comp]:.2f}')

    def _show_symbolic(self):
        from PyQt5.QtWidgets import QDialog, QTextEdit
        from symbolic_calculator import calculate_symbolic_shg_expressions
        state = self._controller.state
        o = state.orientation
        expressions = calculate_symbolic_shg_expressions(state.point_group_name, vars(o))
        dialog = QDialog(self)
        dialog.setWindowTitle('正入射归一化解析式（独立实参数；当前 ZYZ 取向）')
        dialog.resize(900, 650)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText('\n\n'.join(f'{name}:\n{value}' for name, value in expressions.items()))
        layout.addWidget(text)
        dialog.exec_()

    def _show_error(self, message):
        """显示错误"""
        QMessageBox.critical(self, "Error", message)
