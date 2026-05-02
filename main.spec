# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None

# ---------------------------------------------------------------------------
# Submodules that crash PyInstaller's analysis subprocess on Windows.
# Everything else is left to auto-detection.
# ---------------------------------------------------------------------------
_SKIP = {
    'emscripten',           # WebAssembly-only, needs 'js' (not on Windows)
    'onnx.reference',       # segfaults PyInstaller subprocess (0xC0000005)
    'onnx.backend',
    'onnxruntime.quantization',
    'onnxruntime.tools',
    'onnxruntime.training',
}

hiddenimports = []
for pkg in [
    'PyQt5', 'cv2', 'numpy', 'onnxruntime', 'onnx',
    'insightface', 'albumentations', 'scipy',
    'requests', 'urllib3', 'certifi',
    'watchdog', 'aiohttp', 'aiofiles',
    'dotenv', 'sqlalchemy', 'psycopg2',
    'aioboto3', 'aiobotocore', 'botocore', 'boto3',
    'ctypes', 'site',
]:
    try:
        subs = collect_submodules(pkg)
        hiddenimports += [s for s in subs if not any(bad in s for bad in _SKIP)]
    except Exception:
        hiddenimports.append(pkg)

hiddenimports = sorted(set(hiddenimports))

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
datas = []

# Project source folders
for _src, _dst in [('assets','assets'), ('ui','ui'), ('core','core'), ('utils','utils'), ('models','models')]:
    _full = os.path.join(SPECPATH, _src)
    if os.path.isdir(_full):
        datas.append((_full, _dst))

# Package data
for pkg in ['certifi', 'aiohttp', 'onnx', 'onnxruntime', 'sqlalchemy',
            'insightface', 'albumentations', 'botocore', 'aiobotocore']:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

try:
    datas += collect_data_files('PyQt5', include_py_files=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------
binaries = []

for pkg in ['cv2', 'onnxruntime', 'psycopg2', 'numpy', 'scipy']:
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# onnxruntime CUDA provider DLLs (loaded at runtime via LoadLibrary)
try:
    import onnxruntime as _ort
    _capi = os.path.join(os.path.dirname(_ort.__file__), 'capi')
    for _dll in glob.glob(os.path.join(_capi, 'onnxruntime_providers_*.dll')):
        binaries.append((_dll, 'onnxruntime/capi'))
except Exception:
    pass

# cuDNN / CUDA DLLs from  pip install nvidia-cudnn-cu12
try:
    import site as _site
    for _sp in _site.getsitepackages():
        _nvidia = os.path.join(_sp, 'nvidia')
        if os.path.isdir(_nvidia):
            for _pkg in os.listdir(_nvidia):
                _bin = os.path.join(_nvidia, _pkg, 'bin')
                if os.path.isdir(_bin):
                    for _dll in glob.glob(os.path.join(_bin, '*.dll')):
                        binaries.append((_dll, f'nvidia/{_pkg}/bin'))
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
        # Only exclude things that definitively crash analysis AND are unused:
        'onnx.reference',
        'onnx.backend',
        'onnxruntime.quantization',
        'onnxruntime.tools',
        'onnxruntime.training',
        'urllib3.contrib.emscripten',
        # Heavy unused ML frameworks:
        'tensorflow', 'torch', 'torchvision', 'keras',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_version_file = os.path.join(SPECPATH, 'version_info.txt')
_version_file = _version_file if os.path.isfile(_version_file) else None

_icon_file = None
for _c in [os.path.join(SPECPATH, 'assets', 'ownize_logo.ico'),
           os.path.join(SPECPATH, 'assets', 'ownize_logo.png')]:
    if os.path.isfile(_c):
        _icon_file = _c
        break

_no_upx = [
    'vcruntime140.dll', 'msvcp140.dll',
    'Qt5Core.dll', 'Qt5Gui.dll', 'Qt5Widgets.dll', 'qwindows.dll',
    'onnxruntime*.dll', '_pybind_state*.pyd',
    'cudnn*.dll', 'cublas*.dll', 'cufft*.dll', 'cudart*.dll',
]

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='FaceUploaderApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=_no_upx,
    console=False,
    icon=_icon_file,
    version=_version_file,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=True,
    upx_exclude=_no_upx,
    name='FaceUploaderApp v2.1.1',
)
