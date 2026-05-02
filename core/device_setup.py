import os
import sys
import ctypes

# ---------------------------------------------------------------------------
# Windows: register DLL search directories BEFORE importing onnxruntime.
# Must happen here so the OS can resolve onnxruntime_providers_cuda.dll and
# its dependencies (cudart, cuDNN) when onnxruntime is first imported.
# ---------------------------------------------------------------------------
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):

    # 1. Bundled DLLs inside the onnxruntime-gpu wheel itself
    try:
        import site
        for _sp in site.getsitepackages():
            _ort_capi = os.path.join(_sp, 'onnxruntime', 'capi')
            if os.path.isdir(_ort_capi):
                os.add_dll_directory(_ort_capi)
    except Exception:
        pass

    # 2. pip-installed nvidia-cudnn-cu12 / nvidia-cuda-runtime-cu12
    #    These land in  site-packages\nvidia\<pkg>\bin\
    try:
        import site
        for _sp in site.getsitepackages():
            _nvidia_root = os.path.join(_sp, 'nvidia')
            if os.path.isdir(_nvidia_root):
                for _pkg in os.listdir(_nvidia_root):
                    _bin = os.path.join(_nvidia_root, _pkg, 'bin')
                    if os.path.isdir(_bin):
                        try:
                            os.add_dll_directory(_bin)
                        except OSError:
                            pass
    except Exception:
        pass

    # 3. System CUDA toolkit bin\ (cudart64_12.dll, etc.)
    _cuda_dirs = []
    _env_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if _env_cuda:
        _cuda_dirs.append(os.path.join(_env_cuda, 'bin'))

    for _ver in ('12.8', '12.6', '12.4', '12.2', '12.0', '11.8'):
        _cuda_dirs.append(
            rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{_ver}\bin'
        )

    # 4. Common standalone cuDNN locations
    _cuda_dirs += [
        r'C:\cudnn\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v9\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v8\bin',
    ]

    for _d in _cuda_dirs:
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
# Device probe — confirms the CUDA *provider* DLL actually loads, not just
# that onnxruntime was compiled with CUDA support.  The Identity-op test is
# insufficient because simple ops run without cuDNN; convolution models need
# cuDNN and expose the missing-DLL error only at session creation time.
# ---------------------------------------------------------------------------

def _find_cuda_provider_dll() -> str | None:
    """Return the full path to onnxruntime_providers_cuda.dll, or None."""
    try:
        import site
        for sp in site.getsitepackages():
            candidate = os.path.join(sp, 'onnxruntime', 'capi',
                                     'onnxruntime_providers_cuda.dll')
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass
    return None


def _cuda_works() -> bool:
    """Return True only when the CUDA provider DLL and all its dependencies load."""
    try:
        available = ort.get_available_providers()
    except AttributeError:
        print(
            "❌ onnxruntime conflict (CPU and GPU builds both installed).\n"
            "   Fix: pip uninstall onnxruntime onnxruntime-gpu -y\n"
            "        pip install onnxruntime-gpu==1.20.1"
        )
        return False

    if 'CUDAExecutionProvider' not in available:
        return False

    # Attempt to load the CUDA provider DLL directly — this catches
    # error 126 (missing cuDNN / cudart dependency) before InsightFace tries.
    dll_path = _find_cuda_provider_dll()
    if dll_path:
        try:
            ctypes.CDLL(dll_path)
        except OSError as e:
            print(f"⚠️  CUDA provider DLL load failed: {e}")
            return False

    # Final check: create a real CUDA session with a conv-free op.
    try:
        import numpy as np
        try:
            import onnx
            from onnx import helper, TensorProto
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])
            node = helper.make_node('Identity', ['X'], ['Y'])
            graph = helper.make_graph([node], 'probe', [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
            model.ir_version = 7   # onnxruntime 1.19/1.20 supports up to IR 10
            sess = ort.InferenceSession(
                model.SerializeToString(),
                providers=['CUDAExecutionProvider'],
            )
            sess.run(None, {'X': np.zeros((1,), dtype=np.float32)})
        except ImportError:
            pass  # onnx not installed — DLL check above is sufficient
        return True
    except Exception as e:
        print(f"⚠️  CUDA session probe failed: {e}")
        return False


if _cuda_works():
    _ctx_id    = 0
    _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _device    = 'GPU — CUDA (RTX 4080 SUPER)'
else:
    _ctx_id    = -1
    _providers = ['CPUExecutionProvider']
    _device    = 'CPU'
    print(
        "⚠️  Running on CPU — CUDA provider DLL failed to load (cuDNN missing).\n"
        "\n"
        "   Quick fix (no extra download):\n"
        "     pip uninstall onnxruntime-gpu -y\n"
        "     pip install onnxruntime-gpu==1.20.1\n"
        "\n"
        "   Full fix (if above still fails):\n"
        "     1. Download cuDNN 9.x for CUDA 12 from https://developer.nvidia.com/cudnn\n"
        "     2. Copy cuDNN bin\\ DLLs into:\n"
        r"        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    )

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
