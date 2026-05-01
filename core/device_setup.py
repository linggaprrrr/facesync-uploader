import os
import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# ---------------------------------------------------------------------------
# Device probe — try CUDA first, fall back to CPU.
#
# Checking ort.get_available_providers() alone is not enough: the provider
# may be listed but still fail at runtime when cuDNN DLLs are missing.
# We do a real test-session to confirm CUDA actually works before committing.
# ---------------------------------------------------------------------------

def _cuda_works() -> bool:
    """Return True only when a real CUDA inference session can be created."""
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        return False
    try:
        import numpy as np
        # Minimal 1-op ONNX model: Identity(float32 input) → output
        # Built inline so we don't need an external file.
        import struct
        _MINI_ONNX = bytes([
            # Simplified ONNX protobuf for a 1×1 float Identity model
            0x08, 0x07, 0x12, 0x0c, 0x73, 0x65, 0x6e, 0x74, 0x69, 0x6e, 0x65,
            0x6c, 0x5f, 0x76, 0x31, 0x00, 0x3a, 0x2b, 0x0a, 0x09, 0x49, 0x64,
            0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x00, 0x12, 0x01, 0x58, 0x1a,
            0x01, 0x59, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74,
            0x79, 0x5a, 0x0d, 0x0a, 0x01, 0x58, 0x12, 0x08, 0x08, 0x01, 0x12,
            0x04, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x0d, 0x0a, 0x01, 0x59, 0x12,
            0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x00,
        ])
        sess = ort.InferenceSession(
            _MINI_ONNX,
            providers=['CUDAExecutionProvider'],
        )
        sess.run(None, {'X': np.zeros((1, 1), dtype=np.float32)})
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
        "⚠️  CUDA unavailable — running on CPU.\n"
        "    Common fixes:\n"
        "      1. pip uninstall onnxruntime && pip install onnxruntime-gpu>=1.19.0\n"
        "      2. Install cuDNN 9.x for CUDA 12: https://developer.nvidia.com/cudnn\n"
        "      3. Add cuDNN bin/ folder to PATH (e.g. C:\\cudnn\\bin)"
    )

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
