import os
import sys
import ctypes

# ---------------------------------------------------------------------------
# Windows: register DLL search directories BEFORE importing onnxruntime.
# Two cases must be handled:
#   A) Running from source   → DLLs are in site-packages\nvidia\*\bin\
#   B) Running as PyInstaller bundle → DLLs are next to the .exe, under
#      nvidia\<pkg>\bin\  (Windows won't search subdirectories by default)
# ---------------------------------------------------------------------------
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):

    _dll_dirs = []

    if getattr(sys, 'frozen', False):
        # ── Case B: frozen / PyInstaller bundle ──────────────────────────
        _exe_dir = os.path.dirname(sys.executable)

        # onnxruntime capi dir (onnxruntime_providers_cuda.dll lives here)
        _dll_dirs.append(os.path.join(_exe_dir, 'onnxruntime', 'capi'))
        _dll_dirs.append(_exe_dir)

        # nvidia pip DLLs bundled into  <exe_dir>\nvidia\<pkg>\bin\
        _nvidia = os.path.join(_exe_dir, 'nvidia')
        if os.path.isdir(_nvidia):
            for _pkg in os.listdir(_nvidia):
                _dll_dirs.append(os.path.join(_nvidia, _pkg, 'bin'))

    else:
        # ── Case A: running from source / venv ───────────────────────────
        try:
            import site
            for _sp in site.getsitepackages():
                # onnxruntime capi
                _dll_dirs.append(os.path.join(_sp, 'onnxruntime', 'capi'))
                # nvidia pip packages
                _nvidia = os.path.join(_sp, 'nvidia')
                if os.path.isdir(_nvidia):
                    for _pkg in os.listdir(_nvidia):
                        _dll_dirs.append(os.path.join(_nvidia, _pkg, 'bin'))
        except Exception:
            pass

    # System CUDA toolkit (cudart64_12.dll, etc.)
    _env_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if _env_cuda:
        _dll_dirs.append(os.path.join(_env_cuda, 'bin'))
    for _ver in ('12.8', '12.6', '12.4', '12.2', '12.0', '11.8'):
        _dll_dirs.append(
            rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{_ver}\bin'
        )

    # Common standalone cuDNN installs
    _dll_dirs += [
        r'C:\cudnn\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v9\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v8\bin',
    ]

    for _d in _dll_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass

import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')


# ---------------------------------------------------------------------------
# Device probe
# ---------------------------------------------------------------------------

def _find_cuda_provider_dll():
    """Find onnxruntime_providers_cuda.dll in the current environment."""
    candidates = []

    if getattr(sys, 'frozen', False):
        _exe_dir = os.path.dirname(sys.executable)
        candidates.append(
            os.path.join(_exe_dir, 'onnxruntime', 'capi',
                         'onnxruntime_providers_cuda.dll')
        )
    else:
        try:
            import site
            for sp in site.getsitepackages():
                candidates.append(
                    os.path.join(sp, 'onnxruntime', 'capi',
                                 'onnxruntime_providers_cuda.dll')
                )
        except Exception:
            pass

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _cuda_works():
    try:
        available = ort.get_available_providers()
    except AttributeError:
        print("❌ onnxruntime conflict — run: pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu==1.20.1")
        return False

    if 'CUDAExecutionProvider' not in available:
        return False

    dll_path = _find_cuda_provider_dll()
    if dll_path:
        try:
            ctypes.CDLL(dll_path)
        except OSError as e:
            print(f"⚠️  CUDA provider DLL failed: {e}")
            return False

    return True


if _cuda_works():
    _ctx_id    = 0
    _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _device    = 'GPU — CUDA (RTX 4080 SUPER)'
else:
    _ctx_id    = -1
    _providers = ['CPUExecutionProvider']
    _device    = 'CPU'
    print("⚠️  CUDA unavailable — running on CPU.")

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
