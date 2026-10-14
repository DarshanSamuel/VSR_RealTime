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
REM Ensure PyInstaller suppresses onnxruntime from crashing isolated subprocess
python -c "import PyInstaller.building.build_main as m; p=m.__file__; c=open(p, encoding='utf-8').read(); c='suppressed_imports += [\'onnxruntime\']\n' + c if 'suppressed_imports += [\'onnxruntime\']' not in c and (c:=c.replace('suppressed_imports += [\'pyqtgraph.canvas\']', 'suppressed_imports += [\'pyqtgraph.canvas\', \'onnxruntime\']')) else c; open(p, 'w', encoding='utf-8').write(c)" 2>nul

pyinstaller app\vsr_app.spec --noconfirm --distpath dist --workpath build\pyinstaller

REM Remove any conflicting outdated VC runtime DLLs from Qt5\bin that clash with modern Windows/Python 3.14 runtime
if exist "dist\VSR_RealTime\_internal\PyQt5\Qt5\bin\MSVCP140.dll" (
    del /f /q "dist\VSR_RealTime\_internal\PyQt5\Qt5\bin\MSVCP140*.dll" "dist\VSR_RealTime\_internal\PyQt5\Qt5\bin\VCRUNTIME140*.dll" >nul 2>&1
)

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
