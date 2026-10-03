import os
import sys

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

import threading
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# ---------------------------------------------------------------------------
# Face model — loaded lazily. Importing onnxruntime/insightface and building
# CUDA sessions takes seconds (worse on Windows), so it must not run at import
# time. main.py calls warmup_async() so it loads while the login dialog is up.
# ---------------------------------------------------------------------------
_face_app = None
_face_lock = threading.Lock()


def get_face_app():
    """Return the shared FaceAnalysis, loading it on first call (thread-safe)."""
    global _face_app
    with _face_lock:
        if _face_app is None:
            import onnxruntime as ort  # must stay after the PATH setup above
            from insightface.app import FaceAnalysis

            if 'CUDAExecutionProvider' in ort.get_available_providers():
                ctx_id = 0
                # HEURISTIC skips cuDNN's per-conv benchmark on first inference
                providers = [
                    ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}),
                    'CPUExecutionProvider',
                ]
            else:
                ctx_id, providers = -1, ['CPUExecutionProvider']
                print("⚠️  CUDA unavailable — running on CPU.")

            # Only detection (SCRFD-10G) + recognition (ArcFace R100) are used;
            # skipping landmark_2d/3d + genderage saves 3 model loads.
            app = FaceAnalysis(
                name='buffalo_l',
                providers=providers,
                allowed_modules=['detection', 'recognition'],
            )
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            active = app.models['detection'].session.get_providers()[0]
            print(f"✅ InsightFace buffalo_l on {active} — SCRFD-10G + ArcFace R100")
            _face_app = app
    return _face_app


def warmup_async():
    """Start loading the face model in a background thread."""
    threading.Thread(target=get_face_app, name='face-warmup', daemon=True).start()
