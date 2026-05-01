import os
import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# ---------------------------------------------------------------------------
# Device selection: GPU (CUDA device 0) when available, CPU otherwise.
# ctx_id=0  → GPU device 0  (InsightFace uses this to pick the CUDA device)
# ctx_id=-1 → CPU           (InsightFace forces CPU mode)
# ---------------------------------------------------------------------------
_available_providers = ort.get_available_providers()
_cuda_available = 'CUDAExecutionProvider' in _available_providers

if _cuda_available:
    _ctx_id   = 0
    _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _device    = 'GPU (CUDA device 0)'
else:
    _ctx_id   = -1
    _providers = ['CPUExecutionProvider']
    _device    = 'CPU'

face_app = FaceAnalysis(name='buffalo_l', providers=_providers)
face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

print(f"✅ InsightFace buffalo_l on {_device} — SCRFD-10G + ArcFace R100")
