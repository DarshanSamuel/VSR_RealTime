@echo off
REM ───────────────────────────────────────────────
REM  Build VSR Real-Time as a standalone Windows app
REM ───────────────────────────────────────────────
echo.
echo ============================================
echo   Building VSR Real-Time Desktop Application
echo ============================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Install PyInstaller if needed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Building .exe with PyInstaller...
echo.

pyinstaller app\vsr_app.spec --noconfirm --distpath dist --workpath build\pyinstaller

echo.
if exist "dist\VSR_RealTime\VSR_RealTime.exe" (
    echo ============================================
    echo   BUILD SUCCESSFUL!
    echo   Output: dist\VSR_RealTime\VSR_RealTime.exe
    echo ============================================
    echo.
    echo To run: dist\VSR_RealTime\VSR_RealTime.exe
    echo.
    echo NOTE: Place your .onnx model file anywhere
    echo       and browse to it from the app.
) else (
    echo BUILD FAILED - check errors above.
)

pause
