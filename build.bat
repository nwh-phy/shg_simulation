@echo off
setlocal
pushd "%~dp0"
echo Building the SHG analysis workbench in a fresh output directory.
if exist "build\release-venv\Scripts\python.exe" (
    "build\release-venv\Scripts\python.exe" -X utf8 -B tools\build_windows.py
) else (
    python -X utf8 -B tools\build_windows.py
)
set "BUILD_RESULT=%ERRORLEVEL%"
popd
if not "%BUILD_RESULT%"=="0" echo Build failed. See the log path printed above.
pause
exit /b %BUILD_RESULT%
