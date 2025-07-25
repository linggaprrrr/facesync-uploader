# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Icon handling
def get_icon_path():
    png_path = os.path.join(os.getcwd(), 'assets', 'ownize_logo.png')
    ico_path = os.path.join(os.getcwd(), 'assets', 'ownize_logo.ico')
    
    if os.path.exists(ico_path):
        return ico_path
    elif os.path.exists(png_path):
        return png_path
    else:
        return None

# Selective PyTorch collection - only what we actually use
torch_core_modules = [
    'torch',
    'torch._C',
    'torch.nn',
    'torch.nn.functional',
    'torch.nn.modules',
    'torch.nn.modules.linear',
    'torch.nn.modules.conv',
    'torch.nn.modules.pooling',
    'torch.nn.modules.activation',
    'torch.nn.modules.batchnorm',
    'torch.nn.modules.dropout',
    'torch.utils',
    'torch.utils.data',
    'torch.autograd',
    'torch.cuda',
    'torch.serialization',
    'torch.storage',
    'torch.functional',
    'torch.distributed',  # Include this since PyTorch needs it
    'torch._jit_internal',
    'torch.jit',
    'torch.hub',
]

# Essential TorchVision (minimal)
torchvision_core = [
    'torchvision',
    'torchvision.transforms',
    'torchvision.transforms.functional',
    'torchvision.models',
]

# Essential NumPy
numpy_core = [
    'numpy',
    'numpy.core',
    'numpy.core.multiarray',
    'numpy.core.umath',
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.random',
    'numpy.linalg',
]

# Only essential data files
essential_datas = []

# Add assets
if os.path.exists('assets'):
    essential_datas.append(('assets/', 'assets/'))

# Only collect model files, not everything
try:
    # Collect only model files (.pth, .pt) for RetinaFace
    retinaface_data = collect_data_files('retinaface')
    model_files = [(src, dst) for src, dst in retinaface_data 
                   if any(ext in src.lower() for ext in ['.pth', '.pt', '.onnx'])]
    essential_datas.extend(model_files[:3])  # Only first 3 model files
    
    # Same for FaceNet
    facenet_data = collect_data_files('facenet_pytorch') 
    facenet_models = [(src, dst) for src, dst in facenet_data
                      if any(ext in src.lower() for ext in ['.pth', '.pt'])]
    essential_datas.extend(facenet_models[:2])  # Only first 2 model files
    
except Exception as e:
    print(f"⚠️ Model collection warning: {e}")

# All hidden imports
all_hiddenimports = (
    torch_core_modules + 
    torchvision_core + 
    numpy_core + 
    [
        # ML libraries
        'retinaface',
        'facenet_pytorch',
        'onnxruntime',
        'cv2',
        'PIL',
        'PIL.Image',
        
        # PyQt5
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets', 
        'PyQt5.sip',
        
        # Network
        'requests',
        'urllib3',
        'certifi',
        
        # File watching
        'watchdog',
        'watchdog.observers',
        'watchdog.events',
        
        # Utils
        'dotenv',
        'functools',
        'collections',
        'pickle',
        'difflib',
        
        # App modules
        'core',
        'core.watcher',
        'ui',
        'ui.explorer_window',
        'ui.admin_login', 
        'ui.admin_setting',
        'ui.features',
        'utils',
        'utils.face_detector',
    ]
)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=essential_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused modules
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn',
        'jupyter',
        'IPython',
        'notebook',
        'sympy',
        'networkx',
        'bokeh',
        'plotly',
        'seaborn',
        'statsmodels',
        
        # Exclude unused PyTorch components
        'torch.ao',
        'torch.backends.mps',
        'torch.backends.mkl',
        'torch.backends.mkldnn', 
        'torch.fx',
        'torch.package',
        'torch.profiler',
        'torch.quantization',
        'torch.testing',
        'torch.utils.benchmark',
        'torch.utils.bottleneck',
        'torch.utils.tensorboard',
        
        # Exclude unused TorchVision
        'torchvision.datasets',
        'torchvision.io',
        'torchvision.prototype',
        'torchvision.utils',
        
        # Development tools
        'pytest',
        'coverage',
        'pylint',
        'flake8',
        'mypy',
        'setuptools',
        'pip',
        'wheel',
    ],
    noarchive=False,
    optimize=1,
)

# Filter out unnecessary files for faster startup
print("🔄 Optimizing for faster startup...")

# Remove test files and documentation
a.datas = [item for item in a.datas if not any(skip in item[0].lower() 
           for skip in [
               'test', 'tests', '__pycache__', '.pyc', '.pyo',
               'docs/', 'doc/', 'examples/', 'tutorials/',
               'benchmark/', 'profiling/', '.git'
           ])]

# Remove duplicate files
seen = set()
unique_datas = []
for item in a.datas:
    if item[1] not in seen:
        seen.add(item[1])
        unique_datas.append(item)
a.datas = unique_datas

print(f"✅ Optimized to {len(a.datas)} files for faster startup")

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
    strip=False,  # Keep symbols for stability
    upx=False,    # No compression for faster startup
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console for production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=get_icon_path(),
)

app = BUNDLE(
    exe,
    name='FaceSync - Uploader.app',
    icon=get_icon_path(),
    bundle_identifier='com.ownize.facesync.uploader',
    info_plist={
        'CFBundleDisplayName': 'FaceSync - Uploader',
        'CFBundleName': 'FaceSync - Uploader',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.14.0',
    }
)