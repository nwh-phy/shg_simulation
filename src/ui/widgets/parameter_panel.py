"""Chinese parameter controls synchronized from the authoritative simulation state."""
from PyQt5.QtCore import Qt, pyqtSignal, QSignalBlocker
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QComboBox, QDoubleSpinBox, QSlider


class ParameterPanel(QWidget):
    point_group_changed = pyqtSignal(str)
    scan_mode_changed = pyqtSignal(str)
    orientation_changed = pyqtSignal(float, float, float)
    phi_beam_changed = pyqtSignal(float)
    global_scale_changed = pyqtSignal(float)
    parameter_changed = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.group_combo = QComboBox()
        form.addRow('晶体点群', self.group_combo)
        self.components_label = QLabel('')
        self.components_label.setWordWrap(True)
        form.addRow(self.components_label)
        self.scan_mode_combo = QComboBox()
        for label, key in [('偏振角极图', 'alpha_polar'), ('偏振角直角坐标', 'alpha_cartesian'), ('入射角扫描', 'theta_polar')]:
            self.scan_mode_combo.addItem(label, key)
        form.addRow('扫描模式', self.scan_mode_combo)
        self.phi_c_spin = self._spin(form, '晶体 φc (°)', 0, 360)
        self.theta_c_spin = self._spin(form, '晶体 θc (°)', 0, 180)
        self.psi_c_spin = self._spin(form, '晶体 ψc (°)', 0, 360)
        self.theta_fixed_spin = self._spin(form, '固定入射角 (°)', 0, 85)
        self.alpha_spin = self._spin(form, '固定偏振角 (°)', 0, 360)
        self.phi_beam_slider = QSlider(Qt.Horizontal)
        self.phi_beam_slider.setRange(0, 359)
        self.phi_beam_label = QLabel('0°')
        form.addRow('入射面方位角', self.phi_beam_slider)
        form.addRow(self.phi_beam_label)
        self.scale_spin = self._spin(form, '张量整体幅度', 0, 20)
        self.scale_spin.setValue(1)
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(['总横向辐射', '平行通道', '垂直通道'])
        form.addRow('检测方式', self.detection_combo)
        self.input_combo = QComboBox()
        for label, key in [('线偏振', 'linear'), ('p 偏振', 'default'), ('左圆偏振', 'lcp'), ('右圆偏振', 'rcp')]:
            self.input_combo.addItem(label, key)
        form.addRow('入射偏振', self.input_combo)
        note = QLabel('本页为归一化张量模拟。实验反射拟合请使用“实验分析”页。ZYZ：Rz(φc) Ry(θc) Rz(ψc)。')
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.group_combo.currentTextChanged.connect(self.point_group_changed)
        self.scan_mode_combo.currentIndexChanged.connect(lambda _: self.scan_mode_changed.emit(self.scan_mode_combo.currentData()))
        for spin in (self.phi_c_spin, self.theta_c_spin, self.psi_c_spin):
            spin.valueChanged.connect(self._emit_orientation)
        self.phi_beam_slider.valueChanged.connect(lambda v: self.phi_beam_changed.emit(float(v)))
        self.scale_spin.valueChanged.connect(self.global_scale_changed)
        self.theta_fixed_spin.valueChanged.connect(lambda v: self.parameter_changed.emit('theta_fixed', v))
        self.alpha_spin.valueChanged.connect(lambda v: self.parameter_changed.emit('alpha_polarization', v))
        self.detection_combo.currentIndexChanged.connect(lambda v: self.parameter_changed.emit('detection_mode', v))
        self.input_combo.currentIndexChanged.connect(lambda _: self.parameter_changed.emit('input_polarization_mode', self.input_combo.currentData()))

    def _spin(self, form, label, low, high):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        form.addRow(label, spin)
        return spin

    def _emit_orientation(self):
        self.orientation_changed.emit(self.phi_c_spin.value(), self.theta_c_spin.value(), self.psi_c_spin.value())

    def set_point_groups(self, groups):
        self.group_combo.clear()
        self.group_combo.addItems(groups)

    def set_components_text(self, text):
        self.components_label.setText(text)

    def sync_state(self, state):
        widgets = [self.group_combo, self.scan_mode_combo, self.phi_c_spin, self.theta_c_spin, self.psi_c_spin, self.theta_fixed_spin, self.alpha_spin, self.phi_beam_slider, self.scale_spin, self.detection_combo, self.input_combo]
        blockers = [QSignalBlocker(w) for w in widgets]
        self.group_combo.setCurrentText(state.point_group_name)
        self.scan_mode_combo.setCurrentIndex(self.scan_mode_combo.findData(state.scan_mode))
        for spin, value in [(self.phi_c_spin, state.orientation.phi_c), (self.theta_c_spin, state.orientation.theta_c), (self.psi_c_spin, state.orientation.psi_c), (self.theta_fixed_spin, state.theta_fixed), (self.alpha_spin, state.alpha_polarization), (self.scale_spin, state.global_scale)]:
            spin.setValue(value)
        self.phi_beam_slider.setValue(int(state.phi_beam))
        self.phi_beam_label.setText(f'{state.phi_beam:g}°')
        self.detection_combo.setCurrentIndex(state.detection_mode)
        self.input_combo.setCurrentIndex(self.input_combo.findData(state.input_polarization_mode))
        self.components_label.setText('独立分量：' + ', '.join(state.component_values))
        del blockers
