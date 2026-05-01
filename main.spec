# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

# ---------------------------------------------------------------------------
# Hidden imports — pull every sub-package so nothing is missing at runtime
# ---------------------------------------------------------------------------
hiddenimports = []

for pkg in [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtNetwork',
    'PyQt5.sip',
    'cv2',
    'numpy',
    'numpy.core',
    'numpy.lib',
    'numpy.linalg',
    'numpy.random',
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi._pybind_state',
    'insightface',
    'insightface.app',
    'insightface.model_zoo',
    'insightface.utils',
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
    'watchdog.observers.fsevents',   # macOS
    'watchdog.observers.inotify',    # Linux
    'watchdog.observers.winapi',     # Windows
    'aiohttp',
    'aiohttp.web',
    'aiohttp.connector',
    'aiofiles',
    'dotenv',
    'python_dotenv',
    'sqlalchemy',
    'sqlalchemy.dialects',
    'sqlalchemy.dialects.postgresql',
    'sqlalchemy.orm',
    'sqlalchemy.ext',
    'psycopg2',
    'aioboto3',
    'aiobotocore',
    'botocore',
    'boto3',
    'asyncio',
    'logging',
    'logging.handlers',
    'json',
    'pathlib',
    'threading',
    'queue',
    'email',
    'email.mime',
    'email.mime.multipart',
    'email.mime.text',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'pkg_resources',
    'pkg_resources._vendor',
]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        hiddenimports.append(pkg)

# De-duplicate
hiddenimports = sorted(set(hiddenimports))

# ---------------------------------------------------------------------------
# Data files — package assets bundled into the distribution
# ---------------------------------------------------------------------------
datas = [
    ('assets',  'assets'),
    ('ui',      'ui'),
    ('core',    'core'),
    ('utils',   'utils'),
    ('models',  'models'),
]

# Collect data files shipped inside Python packages
for pkg in [
    'certifi',          # CA bundle (requests TLS)
    'aiohttp',
    'onnxruntime',
    'sqlalchemy',
    'insightface',
    'botocore',
    'aiobotocore',
]:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# PyQt5 Qt resource data (translations, etc.)
try:
    datas += collect_data_files('PyQt5', include_py_files=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Binaries — native shared libraries that Python packages depend on
# ---------------------------------------------------------------------------
binaries = []

for pkg in [
    'cv2',
    'onnxruntime',
    'psycopg2',
    'numpy',
]:
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy ML frameworks not used by this app
        'tensorflow',
        'torch',
        'torchvision',
        'keras',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL.ImageTk',   # Tkinter backend — not needed
        'tkinter',
        'test',
        'unittest',
        'pydoc',
        'doctest',
        'difflib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_version_file = 'version_info.txt' if os.path.isfile('version_info.txt') else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceUploaderApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # Never compress these — they break when UPX touches them
        'vcruntime140.dll',
        'msvcp140.dll',
        'Qt5Core.dll',
        'Qt5Gui.dll',
        'Qt5Widgets.dll',
        'qwindows.dll',
        'onnxruntime*.dll',
        '_pybind_state*.pyd',
    ],
    console=False,          # No terminal window on end-user machines
    icon='assets/ownize_logo.ico',
    version=_version_file,  # Windows only: embed version metadata in .exe
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'msvcp140.dll',
        'Qt5Core.dll',
        'Qt5Gui.dll',
        'Qt5Widgets.dll',
        'qwindows.dll',
        'onnxruntime*.dll',
        '_pybind_state*.pyd',
    ],
    name='FaceUploaderApp v2.1.1',
)
