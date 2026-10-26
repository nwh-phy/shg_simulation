import os
import sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import pytest
from PyQt5.QtCore import QEventLoop, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QTableWidget, QPushButton
from ui.workbench import SHGWorkbench
from ui.main_window import NewMainWindow
from ui.controllers.main_controller import MainController
from project_io import export_analysis, load_project


@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_simulator_controls_and_controller_rollback(app):
    window = NewMainWindow()
    try:
        window._param_panel.phi_c_spin.setValue(33)
        assert window._controller.state.orientation.phi_c == 33
        window._param_panel.phi_beam_slider.setValue(61)
        assert window._controller.state.phi_beam == 61
        window._param_panel.scan_mode_combo.setCurrentIndex(1)
        assert window._controller.state.scan_mode == 'alpha_cartesian'
        assert window._canvas.axes.name == 'rectilinear'
        window._reset_btn.click()
        assert window._controller.state.orientation.phi_c == window._param_panel.phi_c_spin.value() == 0
        assert window._param_panel.phi_beam_slider.value() == 0
        assert window._controller.state.scan_mode == 'alpha_polar'
        assert window._canvas.axes.name == 'polar'
        previous = window._controller.state
        # Use a standalone controller for failure tests: no modal UI error dialog.
        controller = MainController(); controller.set_point_group(previous.point_group_name)
        old = controller.state
        errors = []; controller.error_occurred.connect(errors.append)
        assert not controller.set_component_value('bad', 2)
        assert controller.state is old and errors
    finally:
        window.close()


def test_mouse_demo_fit_export_and_project_failure_rollback(app, tmp_path, monkeypatch):
    window = SHGWorkbench()
    try:
        assert not window.group.currentData()
        window.fit_button.click()
        assert '点群' in window.status.text()
        window.action_buttons[3].click()
        window.bootstrap.setValue(20)
        window.starts.setValue(3)
        window.fit_button.click()
        worker = window._worker
        loop = QEventLoop()
        worker.completed.connect(loop.quit); worker.failed.connect(loop.quit)
        QTimer.singleShot(15000, loop.quit)
        loop.exec_(); worker.wait(1000); app.processEvents()
        assert window.result is not None, window.status.text()
        assert window.result['identifiable'] and window.export_button.isEnabled()
        report = export_analysis(tmp_path/'bundle', window.result, window.figure, window._fit_project)
        assert report.exists() and (tmp_path/'bundle'/'fit.pdf').exists()
        document, validated = load_project(tmp_path/'bundle'/'project.shg.json')
        assert len(validated[0]) == 2
        old = window.curves
        bad = tmp_path/'invalid.shg.json'; bad.write_text('{"project_schema":1}', encoding='utf-8')
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **kw: (str(bad), ''))
        window.action_buttons[5].click()
        assert window.curves is old and '保留当前状态' in window.status.text()
        window.incidence.setValue(5)
        assert window.result is None and not window.export_button.isEnabled()
    finally:
        window.close()


def test_quality_dialog_records_explicit_exclusion_without_changing_original(app):
    window = SHGWorkbench(); window.load_demo()
    original = window.curves
    def edit_dialog():
        dialog = window.findChild(QDialog)
        table = dialog.findChild(QTableWidget)
        table.item(0, 6).setCheckState(Qt.Checked)
        table.item(0, 7).setText('测试：人工确认饱和')
        dialog.findChild(QPushButton).click()
    QTimer.singleShot(20, edit_dialog)
    try:
        window.action_buttons[2].click()
        assert window.curves['parallel'].metadata['excluded_indices'] == [0]
        assert window.curves['parallel'].metadata['exclusion_reasons']['0'] == '测试：人工确认饱和'
        assert 'excluded_indices' not in original['parallel'].metadata
    finally:
        window.close()
