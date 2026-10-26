import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog
from ui.experimental_import import ExperimentalImportWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_import_failure_and_changed_settings_preserve_previous_result(app, tmp_path, monkeypatch):
    path = tmp_path / "curve.csv"
    path.write_text("angle_deg,intensity\n0,5\n12,8\n", encoding="utf-8")
    window = ExperimentalImportWindow()
    try:
        assert window.step.text() == ""
        assert not window.step.isEnabled()
        window.path.setText(str(path))
        window.load_button.click()
        previous = window.curve
        assert previous is not None and window.export_button.isEnabled()
        window.path.setText(str(tmp_path / "missing.csv"))
        window.load_button.click()
        assert window.curve is previous
        assert "导入失败" in window.status.text()
        assert not window.export_button.isEnabled()
        window.path.setText(str(path))
        window.load_button.click()
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: str(tmp_path))
        window.export_button.click()
        assert len(list(tmp_path.glob("shg_extract_*/shg_curve.metadata.json"))) == 1
        window.channel.setCurrentIndex(1)
        assert not window.export_button.isEnabled()
    finally:
        window.close()


def test_folder_requires_manual_step_and_does_not_double_angles(app, tmp_path):
    for index in (0, 1):
        (tmp_path / f"sample--{index:03d}--Spec.Data 1.txt").write_text("531 2\n532 8\n533 3\n", encoding="utf-8")
    window = ExperimentalImportWindow()
    try:
        window.mode.setCurrentIndex(2)
        window.path.setText(str(tmp_path))
        window.load_button.click()
        assert window.curve is None
        assert "手动填写" in window.status.text()
        window.step.setText("7.5")
        window.load_button.click()
        assert window.curve.angle_deg.tolist() == [0, 7.5]
        assert window.preview.count() == 2
        window.preview.setCurrentIndex(1)
        assert "7.5" in window.spectrum_ax.get_title()
        previous = window.curve
        window.spectrum_view.setCurrentIndex(1)
        assert window.curve is previous and window.export_button.isEnabled()
        window.target.setValue(900)
        window.load_button.click()
        assert window.curve.angle_deg.tolist() == [0, 7.5]
        assert "超出" in window.status.text()
    finally:
        window.close()
