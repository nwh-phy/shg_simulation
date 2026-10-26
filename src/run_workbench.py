"""Desktop entry and packaged executable verification."""
from pathlib import Path
import json
import os
import sys
import traceback
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMessageBox
from utils.config import configure_matplotlib
from project_io import APP_VERSION


def self_test(directory):
    """Run in the actual frozen runtime using only labeled synthetic inputs."""
    import numpy as np
    from ui.workbench import SHGWorkbench
    from joint_fit import fit_joint, make_demo, FitOptions
    from point_groups import get_all_point_groups
    from project_io import export_analysis, save_project, load_project
    from symbolic_calculator import calculate_symbolic_shg_expressions
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    curves, group, geometry = make_demo()
    result = fit_joint(curves, group, geometry, options=FitOptions(bootstrap=20))
    assert result['identifiable'] and abs(result['parameters']['orientation_deg']-17.3) < 0.25
    window = SHGWorkbench(); window.load_demo()
    window._fit_project = window._project_document(); window._fit_completed(result)
    window.ensurePolished(); window.tabs.widget(1).layout().activate(); QApplication.processEvents(); window.canvas.draw()
    window.grab().save(str(directory/'workbench.png'))
    report = export_analysis(directory/'analysis', result, window.figure, window._fit_project)
    save_project(directory/'roundtrip.shg.json', window._fit_project)
    _, loaded = load_project(directory/'roundtrip.shg.json')
    assert len(loaded[0]) == 2
    expressions = calculate_symbolic_shg_expressions(group)
    assert expressions['I_total'] != 0
    status = {'version': APP_VERSION, 'frozen': bool(getattr(sys, 'frozen', False)), 'point_groups': len(get_all_point_groups()), 'joint_fit_orientation_deg': result['parameters']['orientation_deg'], 'bootstrap_successes': result['bootstrap_successes'], 'project_roundtrip': True, 'symbolic': True, 'report': str(report), 'synthetic_only': True}
    (directory/'self_test.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    window.close()
    return 0


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv); app.setFont(QFont('Microsoft YaHei', 9)); app.setApplicationName('SHGWorkbench'); app.setApplicationVersion(APP_VERSION)
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
    app.setWindowIcon(QIcon(str(root/'IPE_logo.png')))
    configure_matplotlib()
    # Keep plot labels readable inside the dense analysis workbench.
    import matplotlib
    matplotlib.rcParams['font.size'] = 10
    def report_error(kind, value, tb):
        root = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'SHGWorkbench' / 'logs'
        root.mkdir(parents=True, exist_ok=True)
        log = root/'last_error.txt'
        log.write_text(''.join(traceback.format_exception(kind, value, tb)), encoding='utf-8')
        QMessageBox.critical(None, 'SHG 分析错误', f'{value}\n\n错误记录：{log}')
    sys.excepthook = report_error
    if '--self-test' in sys.argv:
        directory = sys.argv[sys.argv.index('--self-test')+1]
        try:
            return self_test(directory)
        except Exception:
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory)/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            return 1
    from ui.workbench import SHGWorkbench
    window = SHGWorkbench(); window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
