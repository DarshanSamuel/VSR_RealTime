# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VSR Real-Time application.

Build command (run from project root):
    pyinstaller app/vsr_app.spec --noconfirm
"""

import os
import sys

block_cipher = None

PROJECT_ROOT = os.path.abspath('.')

a = Analysis(
    ['vsr_app.py'],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # Bundle the config file
        (os.path.join(PROJECT_ROOT, 'configs', 'default.yaml'),
         os.path.join('configs')),
    ],
    hiddenimports=[
        'src.vsr.capture',
        'src.vsr.flow',
        'src.vsr.warp',
        'src.vsr.fusion_classical',
        'src.vsr.utils',
        'src.vsr.infer_onnx',
        'onnxruntime',
        'cv2',
        'yaml',
        'numpy',
    ],
    hookspath=[os.path.join(PROJECT_ROOT, 'app', 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'matplotlib', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VSR_RealTime',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # No console window
    icon=None,          # Set to 'app/icon.ico' if you add an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VSR_RealTime',
)
