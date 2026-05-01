import os
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

API_BASE = os.getenv('BASE_URL', 'https://api.ownize.app')

# InsightFace buffalo_l: SCRFD-10G (face detector) + ArcFace R100 (512-dim embeddings)
# Tries CUDAExecutionProvider first, falls back to CPU automatically.
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

try:
    import onnxruntime as ort
    active_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in active_providers:
        print("✅ InsightFace buffalo_l on GPU (CUDA) — SCRFD-10G + ArcFace R100")
    else:
        print("⚠️ InsightFace buffalo_l on CPU — SCRFD-10G + ArcFace R100")
except Exception:
    print("✅ InsightFace buffalo_l loaded — SCRFD-10G + ArcFace R100")
