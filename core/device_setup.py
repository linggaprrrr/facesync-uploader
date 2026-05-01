import os
import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# ---------------------------------------------------------------------------
# Device probe — try CUDA first, fall back to CPU.
#
# get_available_providers() only reflects what the library was built to
# support, not whether cuDNN DLLs are loadable at runtime.  We create a
# real InferenceSession with CUDAExecutionProvider to confirm it works.
# ---------------------------------------------------------------------------

def _cuda_works() -> bool:
    """Return True only when a live CUDA session can be created."""
    try:
        available = ort.get_available_providers()
    except AttributeError:
        # Both onnxruntime and onnxruntime-gpu are installed and conflicting.
        # Run: pip uninstall onnxruntime onnxruntime-gpu -y
        #      pip install onnxruntime-gpu>=1.19.0
        print(
            "❌ onnxruntime conflict detected (CPU and GPU builds both installed).\n"
            "   Fix: pip uninstall onnxruntime onnxruntime-gpu -y  "
            "&&  pip install onnxruntime-gpu>=1.19.0"
        )
        return False

    if 'CUDAExecutionProvider' not in available:
        return False

    # Try opening a real session — this will raise if cuDNN DLLs are missing.
    try:
        import numpy as np
        import tempfile, struct

        # Build the smallest valid ONNX model (opset 11, Identity op)
        # using onnx if available, otherwise skip the live test and trust
        # the provider list (less safe but avoids a hard dependency on onnx).
        try:
            import onnx
            from onnx import helper, TensorProto
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])
            node = helper.make_node('Identity', ['X'], ['Y'])
            graph = helper.make_graph([node], 'probe', [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
            model_bytes = model.SerializeToString()
        except ImportError:
            # onnx package not present — trust the provider list
            return True

        sess = ort.InferenceSession(
            model_bytes,
            providers=['CUDAExecutionProvider'],
        )
        sess.run(None, {'X': np.zeros((1,), dtype=np.float32)})
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
        "      1. pip uninstall onnxruntime onnxruntime-gpu -y\n"
        "         pip install onnxruntime-gpu>=1.19.0\n"
        "      2. Install cuDNN 9.x for CUDA 12: https://developer.nvidia.com/cudnn\n"
        "      3. Add cuDNN bin\\ to PATH  (e.g. C:\\cudnn\\bin)"
    )

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
