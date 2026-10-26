"""Build to a fresh directory; original data and earlier builds are preserved."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys


def main():
    if sys.platform != 'win32':
        raise SystemExit('Windows EXE must be built on Windows.')
    root = Path(__file__).resolve().parents[1]
    version = '2.0.0'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    work = root/'build'/f'workbench-{version}-{stamp}'
    output = root/'dist'/f'SHGWorkbench-{version}-{stamp}'
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    notices = work/'third_party_licenses'; notices.mkdir()
    shutil.copy2(root/'docs'/'GPL-3.0.txt', notices/'PyQt5-GPL-3.0.txt')
    shutil.copy2(root/'docs'/'LGPL-3.0.txt', notices/'Qt-LGPL-3.0.txt')
    python_license = Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.exists():
        shutil.copy2(python_license, notices/'Python-LICENSE.txt')
    versions = {}
    for package in ['PyQt5', 'PyQt5-Qt5', 'numpy', 'scipy', 'matplotlib', 'sympy', 'mpmath', 'Pillow', 'PyInstaller']:
        distribution = md.distribution(package)
        versions[package] = distribution.version
        for item in distribution.files or []:
            basename = Path(item).name.lower()
            if basename.startswith(('license', 'licence', 'copying', 'copyright')) and Path(item).suffix.lower() not in ('.py', '.pyc', '.dll', '.pyd'):
                source = Path(distribution.locate_file(item))
                if source.is_file():
                    target = notices/package/Path(item).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(source, target)
    (notices/'SOURCES.txt').write_text('Application source and build scripts: https://github.com/nwh-phy/shg_simulation/tree/v2.0.0\nApplication original files retain their MIT license. The combined executable includes GPLv3 PyQt5 and is conveyed with the GPLv3 terms; Qt and other dependencies retain their own licenses.\nPyQt5 5.15.11 source: https://pypi.org/project/PyQt5/5.15.11/#files\nQt 5.15.2 source: https://download.qt.io/archive/qt/5.15/5.15.2/single/\nPython 3.11.9: https://github.com/python/cpython/tree/v3.11.9\nNumPy: https://github.com/numpy/numpy/tree/v1.24.2\nSciPy: https://github.com/scipy/scipy/tree/v1.14.1\nMatplotlib: https://github.com/matplotlib/matplotlib/tree/v3.9.3\nSymPy: https://github.com/sympy/sympy/tree/sympy-1.13.1\nPillow: https://github.com/python-pillow/Pillow/tree/11.0.0\nPyInstaller: https://github.com/pyinstaller/pyinstaller/tree/v6.13.0\nFor exact runtime package versions see build_manifest.json.\n', encoding='utf-8')
    log = work/'pyinstaller.log'
    args = [sys.executable, '-m', 'PyInstaller', '--onefile', '--windowed', '--noupx',
            '--name', f'SHGWorkbench-{version}', '--distpath', str(output), '--workpath', str(work/'work'), '--specpath', str(work),
            '--icon', str(root/'IPE_logo.png'), '--add-data', str(root/'data')+';data', '--add-data', str(root/'docs')+';docs',
            '--add-data', str(root/'IPE_logo.png')+';.',
            '--add-data', str(root/'LICENSE')+';.', '--add-data', str(notices)+';third_party_licenses']
    args += ['--hidden-import', 'matplotlib.backends.backend_pdf']
    for package in ['torch', 'tensorflow', 'pandas', 'IPython', 'notebook', 'ipywidgets', 'PySide2', 'PySide6', 'PyQt6', 'tkinter', 'pytest', 'numba', 'llvmlite', 'xarray', 'dask', 'fsspec', 'jax', 'jaxlib', 'cupy', 'cv2']:
        args += ['--exclude-module', package]
    args.append(str(root/'src'/'run_workbench.py'))
    print('BUILD_LOG', log, flush=True)
    print('OUTPUT_DIRECTORY', output, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-10000:])
        return process.returncode
    executable = output/f'SHGWorkbench-{version}.exe'
    files = [*root.glob('src/**/*.py'), *root.glob('data/*.json'), *root.glob('docs/*.md'), root/'requirements-release.txt']
    manifest = {'version': version, 'built_at_utc': stamp, 'python': platform.python_version(), 'architecture': platform.machine(), 'dependencies': versions,
                'sources_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'executable': executable.name, 'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, capture_output=True, text=True)
    manifest['source_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    manifest['tracked_tree_dirty'] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    (output/'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copytree(notices, output/'third_party_licenses')
    shutil.copy2(root/'LICENSE', output/'LICENSE')
    shutil.copy2(root/'docs'/'使用说明.md', output/'使用说明.txt')
    print('BUILD_COMPLETE', executable, flush=True)
    print('SIZE_MB', round(executable.stat().st_size/1024**2, 2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
