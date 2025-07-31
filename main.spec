# FaceSearchApp.spec

# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules

# Hidden imports untuk tf_keras dan retinaface_pytorch
hiddenimports = (
    collect_submodules('tf_keras') +
    collect_submodules('retinaface_pytorch') +
    collect_submodules('facenet_pytorch')
)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('assets/*', 'assets'),      # Aset gambar/icon/logo
        ('ui/*', 'ui'),              # Folder UI Qt5
        ('core/*', 'core'),          # Folder Core logic
        ('utils/*', 'utils'),        # Folder utils/tools
        
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceUploaderApp 1.1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # True jika ingin jendela terminal muncul
    icon='assets/ownize_logo.ico'  # Gunakan file .ico jika tersedia
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='FaceUploaderApp v1.1.0'
)
