"""Mouse-driven SHG workbench: import, quality review, joint fit, projects, reports."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSignalBlocker
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QScrollArea, QLabel, QPushButton, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QTextBrowser, QTextEdit, QProgressBar, QDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from experimental_data import load_angle_curve
from point_groups import get_all_point_groups, get_components_for_group
from joint_fit import FitOptions, fit_joint, make_demo
from reflection_model import Geometry, OpticalStack, predict
from project_io import APP_VERSION, curve_dict, save_project, load_project, export_analysis, parameter_label
from ui.experimental_import import ExperimentalImportWindow


class FitWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.arguments = arguments

    def run(self):
        try:
            self.completed.emit(fit_joint(**self.arguments, cancelled=self.isInterruptionRequested, progress=self.progress.emit))
        except Exception as error:
            self.failed.emit(str(error))


class SHGWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'SHG 实验分析 {APP_VERSION}')
        self.setFont(QFont('Microsoft YaHei', 9))
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, int(screen.width()*0.94)), min(900, int(screen.height()*0.92)))
        self.curves, self.result, self._worker = {}, None, None
        self._loading = True
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        menu = self.menuBar().addMenu('帮助')
        menu.addAction('版本与许可证', self._licenses)
        self._build_import()
        self._build_analysis()
        from ui.main_window import NewMainWindow
        self.simulator = NewMainWindow()
        self.tabs.addTab(self.simulator.takeCentralWidget(), '3. 张量与解析式')
        help_view = QTextBrowser()
        root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
        manual = root/'docs'/'使用说明.md'
        help_view.setOpenExternalLinks(True)
        help_view.setMarkdown(manual.read_text(encoding='utf-8') if manual.exists() else '先导入两通道数据，明确点群与几何，再开始联合拟合。')
        self.tabs.addTab(help_view, '4. 使用说明')
        self._loading = False
        self._draw()

    def _build_import(self):
        self.importer = ExperimentalImportWindow(self)
        page = QWidget(); layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        for text, channel in [('将当前提取结果用作平行通道 ∥', 'parallel'), ('将当前提取结果用作垂直通道 ⊥', 'perpendicular')]:
            button = QPushButton(text)
            button.clicked.connect(lambda _, c=channel: self._use_extraction(c))
            bar.addWidget(button)
        layout.addLayout(bar)
        layout.addWidget(self.importer.takeCentralWidget())
        self.tabs.addTab(page, '1. 导入与光谱')

    def _build_analysis(self):
        page = QWidget(); outer = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        actions = [('载入 ∥ 两列曲线', lambda: self._load_curve('parallel')), ('载入 ⊥ 两列曲线', lambda: self._load_curve('perpendicular')),
                   ('测量质量与排除', self._quality_dialog), ('载入模拟示例', self.load_demo), ('保存项目', self._save_project), ('打开项目', self._load_project)]
        self.action_buttons = []
        for label, callback in actions:
            button = QPushButton(label); button.clicked.connect(callback); toolbar.addWidget(button); self.action_buttons.append(button)
        outer.addLayout(toolbar)
        self.data_label = QLabel('请载入平行与垂直两组数据。尚未选择材料点群。')
        self.data_label.setWordWrap(True); outer.addWidget(self.data_label)
        body = QHBoxLayout(); outer.addLayout(body, 1)
        left = QWidget(); left.setMinimumWidth(375); left.setMaximumWidth(450)
        left_layout = QVBoxLayout(left)
        self.settings_body = QWidget(); settings_layout = QVBoxLayout(self.settings_body)
        form = QFormLayout()
        settings_layout.addLayout(form)
        self.group = QComboBox(); self.group.addItem('请选择薄片／表面点群', '')
        for group in get_all_point_groups():
            self.group.addItem(group, group)
        form.addRow('材料点群', self.group)
        self.values = QTableWidget(0, 2); self.values.setHorizontalHeaderLabels(['独立分量', '固定值／初值'])
        self.values.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.values.setMaximumHeight(200)
        settings_layout.addWidget(self.values)
        self.fit_ratios = QCheckBox('拟合可观测张量比（高级）')
        self.complex_ratios = QCheckBox('允许复数张量比（实部＋虚部）')
        self.complex_ratios.setEnabled(False)
        self.fit_orientation = QCheckBox('拟合相对实验零点的晶体取向'); self.fit_orientation.setChecked(True)
        self.fit_background = QCheckBox('拟合两个通道各自的常数偏置'); self.fit_background.setChecked(True)
        for widget in (self.fit_ratios, self.complex_ratios, self.fit_orientation, self.fit_background):
            settings_layout.addWidget(widget)
        self.fit_ratios.toggled.connect(self.complex_ratios.setEnabled)
        form2 = QFormLayout(); settings_layout.addLayout(form2)
        self.fixed_phi = self._spin(form2, '固定取向（未勾选拟合时）', -180, 360)
        self.bootstrap = QSpinBox(); self.bootstrap.setRange(0, 1000); self.bootstrap.setValue(100)
        form2.addRow('残差重采样次数', self.bootstrap)
        self.starts = QSpinBox(); self.starts.setRange(1, 72); self.starts.setValue(12); form2.addRow('取向多起点数量', self.starts)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(20261026); form2.addRow('可重复随机种子', self.seed)
        self.ratio_bound = self._spin(form2, '张量比绝对值上限', 0.0001, 1000000, 10)
        self.reference = QComboBox(); self.reference.addItem('自动选择非零可观测分量', ''); form2.addRow('张量比参考分量', self.reference)
        geometry_box = QGroupBox('实验几何与通道响应'); gf = QFormLayout(geometry_box); settings_layout.addWidget(geometry_box)
        self.incidence = self._spin(gf, '入射角（°）', 0, 84.9)
        self.beam_phi = self._spin(gf, '入射面方位（°）', -360, 360)
        self.zero = self._spin(gf, '偏振零点修正（°）', -180, 180)
        self.zero_calibrated = QCheckBox('实验偏振零点已经标定'); gf.addRow(self.zero_calibrated)
        self.analyzer = QComboBox(); self.analyzer.addItems(['同步旋转：∥ / ⊥', '固定检偏器：两个正交通道']); gf.addRow('检偏器模式', self.analyzer)
        self.analyzer_offset = self._spin(gf, '检偏器偏差（°）', -180, 180)
        self.fixed_analyzer = self._spin(gf, '固定检偏器角（°）', -360, 360)
        self.gain = self._spin(gf, '垂直／平行强度响应比', 0.0001, 10000, 1)
        self.gain_calibrated = QCheckBox('已用标定值填写通道响应比'); gf.addRow(self.gain_calibrated)
        optical_box = QGroupBox('反射与基底修正'); of = QFormLayout(optical_box); settings_layout.addWidget(optical_box)
        self.optical_mode = QComboBox(); self.optical_mode.addItems(['相对图样（无光学常数）', '薄片／单一半无限基底', '薄片／介质膜／基底']); of.addRow('光学模型', self.optical_mode)
        self.pump = QLineEdit(); self.pump.setPlaceholderText('修正模式必填，单位 nm'); of.addRow('激发波长', self.pump)
        self.n_sub_w, self.n_sub_2w, self.n_film_w, self.n_film_2w = [QLineEdit() for _ in range(4)]
        for label, widget in [('基底 n+ik（ω）', self.n_sub_w), ('基底 n+ik（2ω）', self.n_sub_2w), ('介质膜 n+ik（ω）', self.n_film_w), ('介质膜 n+ik（2ω）', self.n_film_2w)]:
            widget.setPlaceholderText('按实际波长填写，例如 n+kj'); of.addRow(label, widget)
        self.film_thickness = self._spin(of, '介质膜厚度（nm）', 0, 100000)
        self.constants_source = QLineEdit(); self.constants_source.setPlaceholderText('文献／测量来源；估计值请明确注明'); of.addRow('光学常数来源', self.constants_source)
        scope = QLabel('近正入射、面内电偶极薄片、各向同性基底。高 NA、基底自身 SHG、面外响应不在本模型内。输出相对强度与条件性参数区间。')
        scope.setWordWrap(True); settings_layout.addWidget(scope); settings_layout.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.settings_body); left_layout.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        self.fit_button = QPushButton('开始联合拟合'); self.fit_button.clicked.connect(self.start_fit)
        self.cancel_button = QPushButton('取消'); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_fit)
        buttons.addWidget(self.fit_button); buttons.addWidget(self.cancel_button); left_layout.addLayout(buttons)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); left_layout.addWidget(self.progress)
        self.export_button = QPushButton('导出图、参数、残差和报告…'); self.export_button.setEnabled(False); self.export_button.clicked.connect(self._export); left_layout.addWidget(self.export_button)
        self.status = QLabel('准备就绪。点击“载入模拟示例”可学习完整操作。'); self.status.setWordWrap(True); self.status.setMinimumHeight(65); left_layout.addWidget(self.status)
        body.addWidget(left)
        right = QTabWidget(); body.addWidget(right, 1)
        self.figure = Figure(constrained_layout=True); self.canvas = FigureCanvasQTAgg(self.figure); right.addTab(self.canvas, '数据、图样与残差')
        self.parameters_table = QTableWidget(0, 3); self.parameters_table.setHorizontalHeaderLabels(['参数', '估计值', '条件性 95% 区间'])
        self.parameters_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); right.addTab(self.parameters_table, '参数与可信范围')
        self.notes = QTextEdit(); self.notes.setReadOnly(True); right.addTab(self.notes, '模型、标定与完整结果')
        self.tabs.addTab(page, '2. 联合拟合与报告')
        self.group.currentIndexChanged.connect(self._group_changed)
        self.values.itemChanged.connect(self._dirty)
        for widget in self.settings_body.findChildren(QDoubleSpinBox) + self.settings_body.findChildren(QSpinBox):
            widget.valueChanged.connect(self._dirty)
        for widget in self.settings_body.findChildren(QLineEdit):
            widget.textChanged.connect(self._dirty)
        for widget in self.settings_body.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._dirty)
        for widget in self.settings_body.findChildren(QCheckBox):
            widget.toggled.connect(self._dirty)
        self.optical_mode.currentIndexChanged.connect(self._optical_enabled)
        self._optical_enabled()
        for layout in self.settings_body.findChildren(QFormLayout):
            layout.setRowWrapPolicy(QFormLayout.WrapLongRows)

    def _spin(self, form, label, low, high, value=0):
        widget = QDoubleSpinBox(); widget.setRange(low, high); widget.setDecimals(4); widget.setValue(value); form.addRow(label, widget)
        return widget

    def _licenses(self):
        root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
        paths = sorted((root/'third_party_licenses').rglob('*')) if (root/'third_party_licenses').exists() else [root/'LICENSE', root/'docs'/'GPL-3.0.txt', root/'docs'/'LGPL-3.0.txt']
        paths = [p for p in paths if p.is_file()]
        dialog = QDialog(self); dialog.setWindowTitle(f'SHG 实验分析 {APP_VERSION}：版本与依赖许可证'); dialog.resize(900, 650)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('原项目源码保留 MIT 许可。发布的组合程序包含 GPLv3 PyQt5；完整源代码和构建方法在 GitHub 提供。'))
        selection = QComboBox(); selection.addItems([str(p.relative_to(root)) for p in paths]); layout.addWidget(selection)
        content = QTextEdit(); content.setReadOnly(True); layout.addWidget(content)
        def show(index):
            if index >= 0:
                content.setPlainText(paths[index].read_text(encoding='utf-8', errors='replace'))
        selection.currentIndexChanged.connect(show); show(0); dialog.exec_()

    def _busy(self):
        return self._worker is not None and self._worker.isRunning()

    def _dirty(self, *_):
        if not self._loading:
            self.export_button.setEnabled(False)
            self.result = None
            self.status.setText('输入或设置已改变，请重新拟合。')
            self._draw()

    def _optical_enabled(self, *_):
        corrected, film = self.optical_mode.currentIndex() != 0, self.optical_mode.currentIndex() == 2
        for w in (self.pump, self.n_sub_w, self.n_sub_2w, self.constants_source):
            w.setEnabled(corrected)
        for w in (self.n_film_w, self.n_film_2w, self.film_thickness):
            w.setEnabled(film)

    def _group_changed(self, *_):
        labels = get_components_for_group(self.group.currentData()) if self.group.currentData() else []
        blocker = QSignalBlocker(self.values)
        self.values.setRowCount(len(labels))
        self.values.setMaximumHeight(min(200, 50 + 26*len(labels)))
        self.reference.clear(); self.reference.addItem('自动选择非零可观测分量', '')
        for name in labels:
            self.reference.addItem(name, name)
        for i, label in enumerate(labels):
            item = QTableWidgetItem(label); item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable); self.values.setItem(i, 0, item)
            self.values.setItem(i, 1, QTableWidgetItem('1'))
        del blocker

    def _snapshot(self):
        group = self.group.currentData()
        values = {self.values.item(i, 0).text(): complex(self.values.item(i, 1).text()) for i in range(self.values.rowCount())}
        geometry = Geometry(self.incidence.value(), self.beam_phi.value(), self.zero.value(), ('co_rotating', 'fixed')[self.analyzer.currentIndex()], self.analyzer_offset.value(), self.fixed_analyzer.value(), self.gain.value(), self.gain_calibrated.isChecked(), self.zero_calibrated.isChecked())
        mode = ('relative', 'single', 'film')[self.optical_mode.currentIndex()]
        optics = OpticalStack(mode=mode, pump_nm=float(self.pump.text()) if mode != 'relative' else None,
            substrate_omega=complex(self.n_sub_w.text()) if mode != 'relative' else 1,
            substrate_2omega=complex(self.n_sub_2w.text()) if mode != 'relative' else 1,
            film_omega=complex(self.n_film_w.text()) if mode == 'film' else 1,
            film_2omega=complex(self.n_film_2w.text()) if mode == 'film' else 1,
            film_thickness_nm=self.film_thickness.value() if mode == 'film' else 0, constants_source=self.constants_source.text() if mode != 'relative' else '')
        options = FitOptions(self.fit_ratios.isChecked(), self.complex_ratios.isChecked(), self.fit_background.isChecked(), self.fit_orientation.isChecked(), self.fixed_phi.value(), self.starts.value(), self.bootstrap.value(), self.seed.value(), self.ratio_bound.value(), self.reference.currentData())
        return group, geometry, optics, options, values

    def _error(self, message):
        self.status.setStyleSheet('color: #a11;'); self.status.setText(str(message))

    def _assign_curve(self, channel, curve):
        if self._busy():
            self._error('拟合正在运行，请先取消再更换数据。'); return
        candidate = deepcopy(curve)
        candidate.metadata['channel'] = channel
        self.curves[channel] = candidate
        self._dirty(); self._data_summary(); self._draw()

    def _data_summary(self):
        texts = []
        for name, curve in self.curves.items():
            excluded = len(curve.metadata.get('excluded_indices', []))
            label = '∥' if name == 'parallel' else '⊥'
            synthetic = '【模拟】' if curve.metadata.get('synthetic') else '【实测导入】'
            texts.append(f'{label} {synthetic} {len(curve.angle_deg)} 点，排除 {excluded}；{curve.metadata.get("intensity_unit", "input_units")}')
        self.data_label.setText('；'.join(texts) or '请载入两通道数据。')

    def _use_extraction(self, channel):
        if self.importer.curve is None or not self.importer.export_button.isEnabled():
            self.tabs.setCurrentIndex(1); self._error('请先在光谱页成功提取当前设置的数据。'); return
        self._assign_curve(channel, self.importer.curve); self.tabs.setCurrentIndex(1)

    def _load_curve(self, channel):
        path, _ = QFileDialog.getOpenFileName(self, '选择角度—强度曲线', '', '曲线 (*.csv *.txt *.tsv)')
        if path:
            try:
                self._assign_curve(channel, load_angle_curve(path))
            except Exception as error:
                self._error(f'导入失败，保留已有数据：{error}')

    def load_demo(self):
        if self._busy():
            return
        curves, group, geometry = make_demo()
        self.curves = curves
        self._apply_configuration(group, geometry, OpticalStack(), FitOptions(), {'yyy': 1})
        self._dirty(); self._data_summary(); self.tabs.setCurrentIndex(1)
        self.status.setStyleSheet(''); self.status.setText('已载入模拟 D3h 双通道数据：真实相对取向 17.3°。可点击开始联合拟合。')

    def start_fit(self):
        if self._busy():
            return
        try:
            group, geometry, optics, options, values = self._snapshot()
            if not group:
                raise ValueError('请先选择适合材料薄片／表面的点群；软件不自动判定材料。')
            if set(self.curves) != {'parallel', 'perpendicular'}:
                raise ValueError('请先载入两个通道。')
        except Exception as error:
            self._error(error); return
        self._fit_project = self._project_document()
        self._worker = FitWorker(dict(curves=deepcopy(self.curves), group=group, geometry=geometry, stack=optics, options=options, tensor_values=values), self)
        self._worker.completed.connect(self._fit_completed)
        self._worker.failed.connect(self._fit_failed)
        self._worker.progress.connect(self.progress.setValue)
        self.settings_body.setEnabled(False); self.fit_button.setEnabled(False); self.export_button.setEnabled(False); self.cancel_button.setEnabled(True)
        for b in self.action_buttons:
            b.setEnabled(False)
        self.progress.setValue(0); self.status.setStyleSheet(''); self.status.setText('正在联合拟合与重采样，可取消。')
        self._worker.start()

    def cancel_fit(self):
        if self._busy():
            self._worker.requestInterruption(); self.status.setText('正在停止拟合…')

    def _finish_controls(self):
        self.settings_body.setEnabled(True); self.fit_button.setEnabled(True); self.cancel_button.setEnabled(False)
        for b in self.action_buttons:
            b.setEnabled(True)
        self._optical_enabled()

    def _fit_completed(self, result):
        self.result = result; self._finish_controls(); self.progress.setValue(100); self.export_button.setEnabled(True)
        self.parameters_table.setRowCount(len(result['parameters']))
        for i, (name, value) in enumerate(result['parameters'].items()):
            interval = result['intervals_95'].get(name)
            for col, text in enumerate([parameter_label(name), f'{value:.7g}', f'{interval[0]:.6g} – {interval[1]:.6g}' if interval else '不可辨识／未估计']):
                item = QTableWidgetItem(text); item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable); self.parameters_table.setItem(i, col, item)
        self.notes.setPlainText('\n'.join(result['warnings']) + '\n\n' + json.dumps(result, ensure_ascii=False, indent=2))
        self.status.setStyleSheet('color: #176232;' if result['identifiable'] and not result.get('bounded_solution') else 'color: #9a5b00;')
        self.status.setText(f'完成；成功重采样 {result["bootstrap_successes"]} 次。\n' + ('参数触及边界，未给区间；请调整参考分量或上限。' if result.get('bounded_solution') else '局部参数检查通过；请结合区间与残差判断。' if result['identifiable'] else '参数不能唯一辨识，请减少自由参数。'))
        self._draw()

    def _fit_failed(self, message):
        self._finish_controls(); self.export_button.setEnabled(False); self._error(message)

    def _draw(self):
        if not hasattr(self, 'figure'):
            return
        self.figure.clear()
        polar = [self.figure.add_subplot(221, polar=True), self.figure.add_subplot(222, polar=True)]
        cartesian, residual = self.figure.add_subplot(223), self.figure.add_subplot(224)
        for ax in polar + [cartesian, residual]:
            ax.grid(True, alpha=0.3)
        from matplotlib.ticker import MaxNLocator
        for ax in polar:
            ax.yaxis.set_major_locator(MaxNLocator(4))
        for ax, channel, color in zip(polar, ('parallel', 'perpendicular'), ('#2068ac', '#bd5528')):
            ax.set_title(channel)
            if channel not in self.curves:
                continue
            curve = self.curves[channel]
            keep = np.ones(len(curve.angle_deg), bool); keep[curve.metadata.get('excluded_indices', [])] = False
            ax.scatter(np.deg2rad(curve.angle_deg[keep]), curve.intensity[keep], s=15, color=color)
            cartesian.scatter(curve.angle_deg[keep], curve.intensity[keep], s=16, color=color, label=channel)
            if (~keep).any():
                cartesian.scatter(curve.angle_deg[~keep], curve.intensity[~keep], marker='x', color='gray')
            if self.result:
                r = self.result; geometry = Geometry(**r['geometry']); stack = OpticalStack.from_dict(r['optical_stack'])
                values = {name: complex(*value) for name, value in r['tensor_ratios'].items()}
                angle = np.linspace(min(0, float(min(curve.angle_deg))), max(360, float(max(curve.angle_deg))), 721)
                phi = r['parameters'].get('orientation_deg', r['options']['fixed_orientation_deg'])
                prediction = r['parameters']['amplitude']*predict(r['group'], angle, phi, values, geometry, stack, channel) + r['parameters'].get('background_' + channel, 0)
                ax.plot(np.deg2rad(angle), prediction, color=color, lw=1.4); cartesian.plot(angle, prediction, color=color)
                records = [rec for rec in r['records'] if rec['channel'] == channel]
                residual.scatter([rec['angle_deg'] for rec in records], [rec['residual'] for rec in records], color=color, s=16)
        cartesian.set_xlabel('Incident polarization angle (deg)'); cartesian.set_ylabel('Intensity')
        residual.set_xlabel('Incident polarization angle (deg)'); residual.set_ylabel('Data - model'); residual.axhline(0, color='gray', lw=1)
        if self.curves:
            cartesian.legend(fontsize=8)
        self.canvas.draw_idle()

    def _project_document(self):
        group, geometry, optics, options, values = self._snapshot()
        return {'project_schema': 1, 'app_version': APP_VERSION, 'curves': {name: curve_dict(curve) for name, curve in self.curves.items()},
                'group': group, 'geometry': asdict(geometry), 'optical_stack': optics.to_dict(), 'fit_options': asdict(options), 'tensor_values': {k: str(v) for k, v in values.items()}}

    def _apply_configuration(self, group, geometry, optics, options, values):
        self._loading = True
        self.group.setCurrentIndex(max(0, self.group.findData(group)))
        self._group_changed()
        for i in range(self.values.rowCount()):
            self.values.item(i, 1).setText(str(values.get(self.values.item(i, 0).text(), 1)))
        for widget, value in [(self.incidence, geometry.incidence_deg), (self.beam_phi, geometry.beam_azimuth_deg), (self.zero, geometry.polarization_zero_deg), (self.analyzer_offset, geometry.analyzer_offset_deg), (self.fixed_analyzer, geometry.fixed_analyzer_deg), (self.gain, geometry.perpendicular_gain), (self.fixed_phi, options.fixed_orientation_deg), (self.starts, options.starts), (self.bootstrap, options.bootstrap), (self.seed, options.seed), (self.film_thickness, optics.film_thickness_nm), (self.ratio_bound, options.ratio_bound)]:
            widget.setValue(value)
        self.reference.setCurrentIndex(max(0, self.reference.findData(options.reference_component)))
        for widget, value in [(self.zero_calibrated, geometry.zero_calibrated), (self.gain_calibrated, geometry.gain_calibrated), (self.fit_orientation, options.fit_orientation), (self.fit_ratios, options.fit_ratios), (self.complex_ratios, options.complex_ratios), (self.fit_background, options.fit_background)]:
            widget.setChecked(value)
        self.analyzer.setCurrentIndex(0 if geometry.analyzer_mode == 'co_rotating' else 1)
        self.optical_mode.setCurrentIndex(('relative', 'single', 'film').index(optics.mode))
        self.pump.setText(str(optics.pump_nm) if optics.pump_nm is not None else '')
        for widget, value in [(self.n_sub_w, optics.substrate_omega), (self.n_sub_2w, optics.substrate_2omega), (self.n_film_w, optics.film_omega), (self.n_film_2w, optics.film_2omega)]:
            widget.setText(str(value))
        self.constants_source.setText(optics.constants_source)
        self._loading = False

    def _save_project(self):
        try:
            document = self._project_document()
            path, _ = QFileDialog.getSaveFileName(self, '保存 SHG 项目', 'analysis.shg.json', 'SHG 项目 (*.shg.json)')
            if path:
                save_project(path, document, overwrite=True); self.status.setText(f'项目已保存：{path}')
        except Exception as error:
            self._error(error)

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开 SHG 项目', '', 'SHG 项目 (*.shg.json *.json)')
        if path:
            try:
                document, validated = load_project(path)
                curves, group, geometry, optics, options, values = validated
            except Exception as error:
                self._error(f'项目无效，已保留当前状态：{error}'); return
            self._apply_configuration(group, geometry, optics, options, values)
            self.curves = curves; self._dirty(); self._data_summary(); self._draw(); self.tabs.setCurrentIndex(1)
            self.status.setText('项目已打开，原始文件不需要随分析程序移动；可重新拟合。光谱预览请重新载入源光谱。')

    def _quality_dialog(self):
        if not self.curves or self._busy():
            self._error('请先载入数据，且在拟合结束后编辑。'); return
        dialog = QDialog(self); dialog.setWindowTitle('逐点测量质量；排除需手动勾选'); dialog.resize(1100, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('提示不等于异常结论。只排除手动勾选的记录；原始数据保留。饱和阈值请按探测器实际设置。'))
        rows = [(channel, i, curve) for channel, curve in self.curves.items() for i in range(len(curve.angle_deg))]
        table = QTableWidget(len(rows), 8); table.setHorizontalHeaderLabels(['通道', '原行号', '偏振角', '强度', '估计误差', '质量提示', '排除', '排除原因'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); layout.addWidget(table)
        for row, (channel, i, curve) in enumerate(rows):
            records = curve.metadata.get('records', [])
            rec = records[i] if len(records) == len(curve.angle_deg) else {}
            labels = {'saturated': '达到饱和阈值', 'isolated_spike_review': '孤立峰点，请复核原谱'}
            values = ['平行' if channel == 'parallel' else '垂直', str(i), f'{curve.angle_deg[i]:g}', f'{curve.intensity[i]:.7g}', str(rec.get('uncertainty') or '未估计'), ', '.join(labels.get(flag, flag) for flag in rec.get('quality_flags', [])) or '无自动提示']
            for col, value in enumerate(values):
                item = QTableWidgetItem(value); item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable); table.setItem(row, col, item)
            check = QTableWidgetItem(); check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable); check.setCheckState(Qt.Checked if i in curve.metadata.get('excluded_indices', []) else Qt.Unchecked); table.setItem(row, 6, check)
            table.setItem(row, 7, QTableWidgetItem(curve.metadata.get('exclusion_reasons', {}).get(str(i), '用户手动排除')))
        apply = QPushButton('应用排除选择'); layout.addWidget(apply)
        def commit():
            candidate = deepcopy(self.curves)
            for curve in candidate.values():
                curve.metadata['excluded_indices'] = []; curve.metadata['exclusion_reasons'] = {}
            for row, (channel, i, _) in enumerate(rows):
                if table.item(row, 6).checkState() == Qt.Checked:
                    reason = table.item(row, 7).text().strip()
                    if not reason:
                        self._error('每个排除点需填写原因。'); return
                    candidate[channel].metadata['excluded_indices'].append(i)
                    candidate[channel].metadata['exclusion_reasons'][str(i)] = reason
            self.curves = candidate; self._dirty(); self._data_summary(); self._draw(); dialog.accept()
        apply.clicked.connect(commit); dialog.exec_()

    def _export(self):
        if not self.result or not self.export_button.isEnabled():
            return
        parent = QFileDialog.getExistingDirectory(self, '选择导出位置（创建新的分析文件夹）')
        if parent:
            directory = Path(parent)/datetime.now(timezone.utc).strftime('SHG_analysis_%Y%m%dT%H%M%S_%fZ')
            try:
                report = export_analysis(directory, self.result, self.figure, self._fit_project)
                self.status.setText(f'已导出报告、图与参数：{report}')
            except Exception as error:
                self._error(f'导出失败：{error}')

    def closeEvent(self, event):
        if self._busy():
            self._worker.requestInterruption()
            if not self._worker.wait(3000):
                event.ignore(); self.status.setText('拟合正在停止，稍后再关闭窗口。'); return
        event.accept()
