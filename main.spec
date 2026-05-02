# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob
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
    'urllib3.contrib',
    'certifi',
    'charset_normalizer',
    'idna',
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
    'watchdog.observers.fsevents',
    'watchdog.observers.inotify',
    'watchdog.observers.winapi',
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
    'ctypes',
    'site',
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
        subs = collect_submodules(pkg)
        subs = [s for s in subs if not any(bad in s for bad in (
            'emscripten',           # WebAssembly-only, needs 'js'
            'onnx.reference',       # crashes PyInstaller subprocess (0xC0000005)
            'onnx.backend',         # same
            'onnxruntime.quantization',
            'onnxruntime.tools',
            'onnxruntime.training',
        ))]
        hiddenimports += subs
    except Exception:
        hiddenimports.append(pkg)

hiddenimports = sorted(set(hiddenimports))

# ---------------------------------------------------------------------------
# Data files — package assets bundled into the distribution
# SPECPATH is set by PyInstaller to the directory containing this .spec file.
# ---------------------------------------------------------------------------
datas = []
for _src, _dst in [
    ('assets', 'assets'),
    ('ui',     'ui'),
    ('core',   'core'),
    ('utils',  'utils'),
    ('models', 'models'),
]:
    _full = os.path.join(SPECPATH, _src)
    if os.path.isdir(_full):
        datas.append((_full, _dst))
    else:
        print(f'WARNING: skipping missing folder: {_full}')

for pkg in [
    'certifi',
    'aiohttp',
    'onnx',
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

try:
    datas += collect_data_files('PyQt5', include_py_files=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Binaries — native shared libraries
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

# onnxruntime CUDA provider DLLs (onnxruntime_providers_cuda.dll,
# onnxruntime_providers_shared.dll, onnxruntime_providers_tensorrt.dll)
# These are NOT picked up by collect_dynamic_libs('onnxruntime') because
# they live in capi\ and are loaded at runtime via LoadLibrary.
try:
    import onnxruntime as _ort
    _ort_capi = os.path.join(os.path.dirname(_ort.__file__), 'capi')
    for _dll in glob.glob(os.path.join(_ort_capi, 'onnxruntime_providers_*.dll')):
        binaries.append((_dll, 'onnxruntime/capi'))
except Exception:
    pass

# cuDNN + CUDA runtime DLLs installed via  pip install nvidia-cudnn-cu12
# They land in  site-packages\nvidia\<pkg>\bin\*.dll
try:
    import site as _site
    for _sp in _site.getsitepackages():
        _nvidia_root = os.path.join(_sp, 'nvidia')
        if os.path.isdir(_nvidia_root):
            for _pkg_name in os.listdir(_nvidia_root):
                _bin_dir = os.path.join(_nvidia_root, _pkg_name, 'bin')
                if os.path.isdir(_bin_dir):
                    for _dll in glob.glob(os.path.join(_bin_dir, '*.dll')):
                        binaries.append((_dll, f'nvidia/{_pkg_name}/bin'))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [os.path.join(SPECPATH, 'main.py')],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # WebAssembly-only — requires 'js' module that doesn't exist on Windows
        'urllib3.contrib.emscripten',
        # onnx.reference crashes PyInstaller's analysis subprocess (0xC0000005).
        # Exclude only the heavy optional sub-packages — insightface needs
        # the core onnx package at runtime so we must keep it.
        'onnx.reference',
        'onnx.backend',
        'onnxruntime.quantization',
        'onnxruntime.tools',
        'onnxruntime.training',
        # Heavy ML frameworks not used by this app
        'tensorflow',
        'torch',
        'torchvision',
        'keras',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL.ImageTk',
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

_version_file = os.path.join(SPECPATH, 'version_info.txt')
_version_file = _version_file if os.path.isfile(_version_file) else None

# Prefer .ico, fall back to .png (PyInstaller auto-converts), then no icon.
_icon_file = None
for _icon_candidate in (
    os.path.join(SPECPATH, 'assets', 'ownize_logo.ico'),
    os.path.join(SPECPATH, 'assets', 'ownize_logo.png'),
    os.path.join(SPECPATH, 'assets', 'ownize_logo_2.png'),
):
    if os.path.isfile(_icon_candidate):
        _icon_file = _icon_candidate
        break

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
        # Never UPX-compress these — they break
        'vcruntime140.dll',
        'msvcp140.dll',
        'Qt5Core.dll',
        'Qt5Gui.dll',
        'Qt5Widgets.dll',
        'qwindows.dll',
        'onnxruntime*.dll',
        '_pybind_state*.pyd',
        'cudnn*.dll',
        'cublas*.dll',
        'cufft*.dll',
        'cudart*.dll',
        'nvinfer*.dll',
    ],
    console=False,
    icon=_icon_file,
    version=_version_file,
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
        'cudnn*.dll',
        'cublas*.dll',
        'cufft*.dll',
        'cudart*.dll',
        'nvinfer*.dll',
    ],
    name='FaceUploaderApp v2.1.1',
)
