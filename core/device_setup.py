import os
import sys
import ctypes

# ---------------------------------------------------------------------------
# Windows: make bundled CUDA/cuDNN DLLs findable BEFORE importing onnxruntime.
#
# IMPORTANT: os.add_dll_directory() only helps Python's own loader (.pyd files).
# onnxruntime's C++ code calls LoadLibrary("cudnn64_9.dll") internally using
# the standard Windows search order, which checks PATH — not add_dll_directory.
# So we MUST prepend our nvidia\*\bin dirs to PATH as well.
# ---------------------------------------------------------------------------
if sys.platform == 'win32':

    _nvidia_bins = []

    if getattr(sys, 'frozen', False):
        # PyInstaller 6.x: all files are under _internal\ (sys._MEIPASS)
        _internal = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))

        # onnxruntime provider DLLs
        _nvidia_bins.append(os.path.join(_internal, 'onnxruntime', 'capi'))
        _nvidia_bins.append(_internal)

        # nvidia pip DLLs: cudnn, cublas, cudart, etc.
        _nvidia_root = os.path.join(_internal, 'nvidia')
        if os.path.isdir(_nvidia_root):
            for _pkg in os.listdir(_nvidia_root):
                _bin = os.path.join(_nvidia_root, _pkg, 'bin')
                if os.path.isdir(_bin):
                    _nvidia_bins.append(_bin)
    else:
        try:
            import site
            for _sp in site.getsitepackages():
                _nvidia_bins.append(os.path.join(_sp, 'onnxruntime', 'capi'))
                _nvidia_root = os.path.join(_sp, 'nvidia')
                if os.path.isdir(_nvidia_root):
                    for _pkg in os.listdir(_nvidia_root):
                        _bin = os.path.join(_nvidia_root, _pkg, 'bin')
                        if os.path.isdir(_bin):
                            _nvidia_bins.append(_bin)
        except Exception:
            pass

    # System CUDA toolkit — appended AFTER bundled dirs so bundled wins
    _env_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if _env_cuda:
        _nvidia_bins.append(os.path.join(_env_cuda, 'bin'))
    for _ver in ('12.8', '12.6', '12.4', '12.2', '12.0', '11.8'):
        _nvidia_bins.append(
            rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{_ver}\bin'
        )

    # Filter to existing dirs only
    _valid_bins = [d for d in _nvidia_bins if os.path.isdir(d)]

    if hasattr(os, 'add_dll_directory'):
        for _d in _valid_bins:
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass

    # Prepend to PATH so onnxruntime's internal LoadLibrary calls find our DLLs.
    # This is the critical step — without it, onnxruntime searches only system PATH.
    _extra = os.pathsep.join(_valid_bins)
    os.environ['PATH'] = _extra + os.pathsep + os.environ.get('PATH', '')

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
        _internal = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        candidates.append(
            os.path.join(_internal, 'onnxruntime', 'capi',
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
