@echo off

echo =================================================================
echo          SHG Simulator PyInstaller Build Script
echo =================================================================
echo.
echo This script will package the Python application into a single .exe file.
echo It assumes you are running it from the project root directory.
echo.
echo Required files:
echo   - IPE_logo.png (in the project root)
echo   - data/point_group_data.json (ASSUMPTION: please verify this file exists)
echo.

REM Set the name for the output executable
set "APP_NAME=SHG_Simulator"

REM Set the path to the main script
set "MAIN_SCRIPT=src/main.py"

REM Define resource files to be added. Format is "SOURCE;DESTINATION_IN_EXE"
set "LOGO_FILE=IPE_logo.png"
set "JSON_DATA_FILE=data/point_group_data.json"

echo Checking for required files...
IF NOT EXIST "%LOGO_FILE%" (
    echo [ERROR] Logo file not found at: %LOGO_FILE%
    pause
    exit /b 1
)
IF NOT EXIST "%JSON_DATA_FILE%" (
    echo [WARNING] Data file not found at: %JSON_DATA_FILE%
    echo This might be okay if the file is not needed, but please double check.
)
echo.

REM Activate virtual environment if it exists
IF EXIST venv\Scripts\activate.bat (
    echo "Activating virtual environment 'venv'..."
    call venv\Scripts\activate.bat
)

echo "Starting PyInstaller..."
echo.

pyinstaller --noconfirm --onefile --windowed --name %APP_NAME% ^
    --add-data "%LOGO_FILE%;." ^
    --add-data "%JSON_DATA_FILE%;data" ^
    --icon="%LOGO_FILE%" ^
    --exclude-module "torch" ^
    --exclude-module "tensorflow" ^
    --exclude-module "pandas" ^
    --exclude-module "numba" ^
    %MAIN_SCRIPT%
    
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] PyInstaller failed. Please check the output above.
    pause
    exit /b 1
)

echo.
echo =================================================================
echo Build successful!
echo The executable can be found in the 'dist' folder:
echo   dist\%APP_NAME%.exe
echo =================================================================
echo.

pause 