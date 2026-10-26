"""Independent experimental import window shared by both simulator entry points."""
from datetime import datetime, timezone
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QDoubleSpinBox, QPushButton, QFileDialog, QScrollArea,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from experimental_data import (
    ExtractionSettings, load_angle_curve, load_spectral_table,
    load_spectrum_folder, extract_curve, export_curve,
)


class ExperimentalImportWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Microsoft YaHei", 9))
        self.curve = None
        self.setWindowTitle("SHG 实验数据导入与提取")
        self.resize(1250, 800)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        controls = QWidget()
        form_layout = QVBoxLayout(controls)
        note = QLabel("反射测量；角度为已换算的入射偏振角 α（°）。\n导入时不乘二、不取模、不补闭合点。")
        note.setWordWrap(True)
        form_layout.addWidget(note)
        form = QFormLayout()
        form_layout.addLayout(form)

        self.mode = QComboBox()
        self.mode.addItems(["两列：角度、强度", "三列：角度、波长、强度", "目录：每个序号一条光谱"])
        form.addRow("输入形式", self.mode)
        self.path = QLineEdit()
        self.path.setPlaceholderText("选择文件或单个测量序列目录")
        form.addRow("输入路径", self.path)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        form.addRow(browse)
        self.channel = QComboBox()
        self.channel.addItems(["沿用文件标注（无则未标注）", "平行 ∥", "垂直 ⊥"])
        form.addRow("检测通道", self.channel)
        self.step = QLineEdit()
        self.step.setPlaceholderText("必填；允许负步长，如 12 或 -12")
        form.addRow("偏振角步长（°）", self.step)
        self.start = QDoubleSpinBox()
        self.start.setRange(-10000, 10000)
        self.start.setDecimals(4)
        form.addRow("序号 000 起始角（°）", self.start)

        self.target = QDoubleSpinBox()
        self.target.setRange(1, 100000)
        self.target.setDecimals(4)
        self.target.setValue(532)
        self.target.setToolTip("532 nm 仅来自旧分析的建议值，请按本次实验的 2ω 波长设置。")
        form.addRow("目标 2ω 波长（nm）", self.target)
        self.method = QComboBox()
        self.method.addItems(["最近实测波长点", "目标波长处线性插值", "信号窗口内最大值", "扣背景后积分面积（强度×nm）"])
        form.addRow("提取方法", self.method)
        self.half_width = QDoubleSpinBox()
        self.half_width.setRange(0.0001, 10000)
        self.half_width.setDecimals(4)
        self.half_width.setValue(1)
        form.addRow("信号窗口半宽（nm）", self.half_width)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.0001, 10000)
        self.tolerance.setDecimals(4)
        self.tolerance.setValue(0.5)
        form.addRow("采样点允许偏差（nm）", self.tolerance)
        self.background = QLineEdit()
        self.background.setPlaceholderText("留空不扣背景；如 510:525;539:560")
        form.addRow("背景波段（nm）", self.background)
        self.background_model = QComboBox()
        self.background_model.addItems(['常数背景', '线性背景'])
        form.addRow('背景模型', self.background_model)
        self.exposure = QLineEdit()
        self.exposure.setPlaceholderText('可空；填写则除以曝光时间')
        form.addRow('曝光时间（秒）', self.exposure)
        self.power = QLineEdit()
        self.power.setPlaceholderText('可空；填写则除以功率平方')
        form.addRow('激发功率（mW）', self.power)
        self.saturation = QLineEdit()
        self.saturation.setPlaceholderText('可空；按探测器实际满量程设置')
        form.addRow('饱和阈值（原始强度）', self.saturation)
        self.preview = QComboBox()
        self.preview.currentIndexChanged.connect(self._draw)
        form.addRow("查看原始光谱", self.preview)
        self.spectrum_view = QComboBox()
        self.spectrum_view.addItems(["目标波长附近", "完整光谱"])
        self.spectrum_view.currentIndexChanged.connect(self._draw)
        form.addRow("光谱显示范围", self.spectrum_view)

        self.load_button = QPushButton("导入 / 提取并预览")
        self.load_button.clicked.connect(self.load_data)
        form_layout.addWidget(self.load_button)
        self.export_button = QPushButton("导出两列 CSV 与处理记录…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        form_layout.addWidget(self.export_button)
        self.status = QLabel("请先选择输入数据。目录模式必须手动填写步长。")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addWidget(self.status)
        form_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(410)
        scroll.setMaximumWidth(460)
        scroll.setWidget(controls)
        layout.addWidget(scroll)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.angle_ax, self.spectrum_ax = self.figure.subplots(2, 1)
        layout.addWidget(self.canvas, 1)

        for widget in (self.path, self.step, self.background, self.exposure, self.power, self.saturation):
            widget.textChanged.connect(self._mark_dirty)
        for widget in (self.start, self.target, self.half_width, self.tolerance):
            widget.valueChanged.connect(self._mark_dirty)
        for widget in (self.mode, self.method, self.channel, self.background_model):
            widget.currentIndexChanged.connect(self._mark_dirty)
        self.mode.currentIndexChanged.connect(self._update_enabled)
        self.method.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()
        self._draw()

    def _update_enabled(self, *_):
        raw = self.mode.currentIndex() != 0
        folder = self.mode.currentIndex() == 2
        self.step.setEnabled(folder)
        self.start.setEnabled(folder)
        for widget in (self.target, self.method, self.background, self.background_model, self.exposure, self.power, self.saturation):
            widget.setEnabled(raw)
        self.half_width.setEnabled(raw and self.method.currentIndex() in (2, 3))
        self.tolerance.setEnabled(raw and self.method.currentIndex() != 2)

    def _mark_dirty(self, *_):
        self.export_button.setEnabled(False)
        if self.curve is not None:
            self.status.setText("参数已改变，请重新导入 / 提取。图中仍为上次成功的结果。")

    def _browse(self):
        if self.mode.currentIndex() == 2:
            selected = QFileDialog.getExistingDirectory(self, "选择单个测量序列目录", self.path.text())
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "选择实验数据", self.path.text(), "文本数据 (*.csv *.txt *.tsv);;所有文件 (*)")
        if selected:
            self.path.setText(selected)

    def _settings(self):
        ranges = []
        if self.background.text().strip():
            try:
                for item in self.background.text().replace("；", ";").split(";"):
                    low, high = item.strip().split(":")
                    ranges.append((float(low), float(high)))
            except ValueError:
                raise ValueError("背景波段格式应为 下限:上限，多段用分号分隔。") from None
        return ExtractionSettings(
            target_nm=self.target.value(),
            method=("nearest", "linear", "window_max", "integral")[self.method.currentIndex()],
            half_width_nm=self.half_width.value(),
            max_distance_nm=self.tolerance.value(),
            background_ranges_nm=tuple(ranges),
            background_model=('constant', 'linear')[self.background_model.currentIndex()],
            exposure_s=float(self.exposure.text()) if self.exposure.text().strip() else None,
            power_mw=float(self.power.text()) if self.power.text().strip() else None,
            saturation_level=float(self.saturation.text()) if self.saturation.text().strip() else None,
        )

    def load_data(self):
        try:
            if not self.path.text().strip():
                raise ValueError("请先选择输入文件或目录。")
            if self.mode.currentIndex() == 0:
                candidate = load_angle_curve(self.path.text())
            else:
                settings = self._settings()
                if self.mode.currentIndex() == 1:
                    spectra, metadata = load_spectral_table(self.path.text())
                else:
                    if not self.step.text().strip():
                        raise ValueError("请手动填写本次测量的偏振角步长；没有默认值。")
                    spectra, metadata = load_spectrum_folder(self.path.text(), float(self.step.text()), self.start.value())
                candidate = extract_curve(spectra, settings, metadata)
            if self.channel.currentIndex():
                candidate.metadata["channel"] = ("unspecified", "parallel", "perpendicular")[self.channel.currentIndex()]
        except (OSError, ValueError, TypeError) as error:
            self.status.setText(f"导入失败：{error}\n已保留上次成功的结果。")
            self.export_button.setEnabled(False)
            return
        self.curve = candidate
        self.preview.blockSignals(True)
        self.preview.clear()
        self.preview.addItems([f"{i + 1}: α={s.angle_deg:g}°  {Path(s.source).name}" for i, s in enumerate(candidate.spectra)])
        self.preview.blockSignals(False)
        self.preview.setEnabled(bool(candidate.spectra))
        self._draw()
        warnings = "\n".join(candidate.metadata.get("warnings", []))
        self.status.setText(f"成功：{len(candidate.angle_deg)} 条测量记录。\n原始角度与重复测量均保留。\n{warnings}")
        self.export_button.setEnabled(True)

    def _draw(self, *_):
        for ax in (self.angle_ax, self.spectrum_ax):
            ax.clear()
            ax.grid(True, alpha=0.25)
        self.angle_ax.set_xlabel("Incident polarization angle α (deg)")
        self.angle_ax.set_ylabel("Intensity (input units)")
        self.spectrum_ax.set_xlabel("Wavelength (nm)")
        self.spectrum_ax.set_ylabel("Raw intensity (input units)")
        if self.curve is not None:
            self.angle_ax.plot(self.curve.angle_deg, self.curve.intensity, "o", markersize=5)
            if self.curve.spectra:
                index = max(0, self.preview.currentIndex())
                spectrum = self.curve.spectra[index]
                record = self.curve.metadata["records"][index]
                settings = self.curve.metadata["extraction"]
                self.spectrum_ax.plot(spectrum.wavelength_nm, spectrum.intensity, lw=1)
                self.spectrum_ax.plot(record["sample_wavelength_nm"], record.get('preview_raw_intensity', record['raw_intensity']), "ro")
                self.spectrum_ax.axvline(settings["target_nm"], color="red", ls="--", alpha=0.5)
                if settings["background_ranges_nm"]:
                    beta = record.get('baseline_coefficients', [record['background']])
                    baseline = beta[0] + (beta[1]*(spectrum.wavelength_nm-settings['target_nm']) if len(beta) == 2 else 0*spectrum.wavelength_nm)
                    self.spectrum_ax.plot(spectrum.wavelength_nm, baseline, color='green', ls=':')
                if settings["method"] in ("window_max", "integral"):
                    self.spectrum_ax.axvspan(settings["target_nm"] - settings["half_width_nm"], settings["target_nm"] + settings["half_width_nm"], alpha=0.15, color="red")
                for low, high in settings["background_ranges_nm"]:
                    self.spectrum_ax.axvspan(low, high, alpha=0.15, color="green")
                if self.spectrum_view.currentIndex() == 0:
                    import numpy as np
                    low = settings["target_nm"] - max(5, settings["half_width_nm"] * 2)
                    high = settings["target_nm"] + max(5, settings["half_width_nm"] * 2)
                    for bg_low, bg_high in settings["background_ranges_nm"]:
                        low, high = min(low, bg_low), max(high, bg_high)
                    low, high = max(low, spectrum.wavelength_nm[0]), min(high, spectrum.wavelength_nm[-1])
                    self.spectrum_ax.set_xlim(low, high)
                    local = spectrum.intensity[(spectrum.wavelength_nm >= low) & (spectrum.wavelength_nm <= high)]
                    if len(local):
                        minimum, maximum = float(np.min(local)), float(np.max(local))
                        margin = max((maximum - minimum) * 0.1, abs(maximum) * 0.001, 0.01)
                        self.spectrum_ax.set_ylim(minimum - margin, maximum + margin)
                self.spectrum_ax.set_title(f"α={spectrum.angle_deg:g}°; selected λ={record['sample_wavelength_nm']:.4f} nm")
            else:
                self.spectrum_ax.text(0.5, 0.5, "Two-column input contains no raw spectra", ha="center", transform=self.spectrum_ax.transAxes)
        self.canvas.draw_idle()

    def _export(self):
        if self.curve is None or not self.export_button.isEnabled():
            return
        parent = QFileDialog.getExistingDirectory(self, "选择导出位置（将创建新文件夹）", str(Path.cwd()))
        if parent:
            directory = Path(parent) / datetime.now(timezone.utc).strftime("shg_extract_%Y%m%dT%H%M%S_%fZ")
            try:
                csv_path, _ = export_curve(self.curve, directory)
            except (OSError, ValueError, TypeError) as error:
                self.status.setText(f"导出失败：{error}")
                return
            self.status.setText(f"已导出：{csv_path}\n同目录保存 metadata.json，原始输入未修改。")


def show_experimental_import(parent):
    window = getattr(parent, "_experimental_import_window", None)
    if window is None:
        window = ExperimentalImportWindow(parent)
        parent._experimental_import_window = window
    window.show()
    window.raise_()
    return window
