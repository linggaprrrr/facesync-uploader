# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# Simple icon handling
def get_icon_path():
    png_path = os.path.join(os.getcwd(), 'assets', 'ownize_logo.png')
    ico_path = os.path.join(os.getcwd(), 'assets', 'ownize_logo.ico')
    
    if os.path.exists(ico_path):
        return ico_path
    elif os.path.exists(png_path):
        return png_path
    else:
        return None

# Collect all necessary modules without filtering
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

# Essential app data
app_datas = []
if os.path.exists('assets'):
    app_datas.append(('assets/', 'assets/'))

# Combine all data
all_datas = torch_datas + numpy_datas + cv2_datas + app_datas
all_binaries = torch_binaries + numpy_binaries + cv2_binaries  
all_hiddenimports = torch_hiddenimports + numpy_hiddenimports + cv2_hiddenimports

# Add essential hidden imports
all_hiddenimports.extend([
    'retinaface',
    'facenet_pytorch', 
    'onnxruntime',
    'PyQt5.sip',
    'requests',
    'urllib3',
    'certifi',
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
    'dotenv',
    'PIL',
    'PIL.Image',
    'core',
    'core.watcher',
    'ui',
    'ui.explorer_window', 
    'ui.admin_login',
    'ui.admin_setting',
    'ui.features',
    'utils',
    'utils.face_detector',
])

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only exclude modules that definitely aren't needed
        'tkinter',
        'matplotlib.pyplot',
        'pandas', 
        'jupyter',
        'IPython',
        'notebook',
    ],
    noarchive=False,
    optimize=0,  # No optimization to avoid issues
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaceSync - Uploader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip
    upx=True,    # Don't compress
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=get_icon_path(),
)

# macOS bundle
app = BUNDLE(
    exe,
    name='FaceSync - Uploader.app',
    icon=get_icon_path(),
    bundle_identifier='com.ownize.facesync.uploader',
)