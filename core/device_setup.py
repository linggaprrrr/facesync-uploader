import os
import sys

# ---------------------------------------------------------------------------
# Windows: register CUDA DLL directories BEFORE importing onnxruntime.
# onnxruntime-gpu needs cudart64_12.dll (and friends) to be discoverable;
# they live in the CUDA toolkit's bin\ folder which is not always on PATH.
# os.add_dll_directory() was added in Python 3.8 and is Windows-only.
# ---------------------------------------------------------------------------
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _cuda_candidates = [
        # CUDA toolkit default install paths (12.x, 11.x)
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
        # cuDNN standalone install (common locations)
        r'C:\cudnn\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v9\bin',
        r'C:\Program Files\NVIDIA\CUDNN\v8\bin',
    ]
    # Also honour a CUDA_PATH env-var set by the toolkit installer
    _env_cuda = os.environ.get('CUDA_PATH')
    if _env_cuda:
        _cuda_candidates.insert(0, os.path.join(_env_cuda, 'bin'))

    for _path in _cuda_candidates:
        if os.path.isdir(_path):
            try:
                os.add_dll_directory(_path)
            except OSError:
                pass

import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# ---------------------------------------------------------------------------
# Device probe — try CUDA first, fall back to CPU.
# ---------------------------------------------------------------------------

def _cuda_works() -> bool:
    """Return True only when a live CUDA session can be created."""
    try:
        available = ort.get_available_providers()
    except AttributeError:
        print(
            "❌ onnxruntime conflict (CPU and GPU builds both installed).\n"
            "   Fix: pip uninstall onnxruntime onnxruntime-gpu -y\n"
            "        pip install onnxruntime-gpu==1.19.2"
        )
        return False

    if 'CUDAExecutionProvider' not in available:
        return False

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
            # Cap IR version to 7 — onnxruntime 1.19.x supports up to IR 10,
            # but recent onnx packages emit IR 13 by default.
            model.ir_version = 7
            model_bytes = model.SerializeToString()
            sess = ort.InferenceSession(model_bytes, providers=['CUDAExecutionProvider'])
            sess.run(None, {'X': np.zeros((1,), dtype=np.float32)})
        except ImportError:
            # onnx package absent — trust the provider list
            pass
        return True
    except Exception as e:
        print(f"⚠️  CUDA probe failed: {e}")
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
        "⚠️  Running on CPU.  To enable GPU:\n"
        "      pip uninstall onnxruntime-gpu -y\n"
        "      pip install onnxruntime-gpu==1.19.2\n"
        "   If that still fails, add the CUDA bin\\ folder to your system PATH."
    )

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
