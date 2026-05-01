# utils/face_detector.py
# InsightFace buffalo_l: SCRFD-10G (detection) + ArcFace R100 (512-dim embedding)

import cv2
import numpy as np
import logging
import time
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.device_setup import face_app, API_BASE

logger = logging.getLogger(__name__)

# Global lock — InsightFace FaceAnalysis is not thread-safe
_inference_lock = threading.Lock()


def process_faces_in_image_optimized(
    file_path: str,
    is_selfie_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Detect faces and generate 512-dim ArcFace R100 embeddings in one pass.

    Returns a list of dicts:
        {
            'embedding': [512-dim float],
            'bbox':      {'x': int, 'y': int, 'w': int, 'h': int},
            'confidence': float  (0–1, from SCRFD-10G)
        }
    """
    start_time = time.time()
    try:
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"❌ Cannot load image: {file_path}")
            return []

        # InsightFace expects BGR (same as OpenCV default)
        with _inference_lock:
            faces = face_app.get(img)

        if not faces:
            logger.warning(f"⚠️ No faces detected: {Path(file_path).name}")
            return []

        # Reference mode: keep only the largest face
        if is_selfie_mode and len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )[:1]

        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            results.append({
                'embedding': face.embedding.tolist(),   # 512-dim ArcFace R100
                'bbox': {'x': x1, 'y': y1, 'w': w, 'h': h},
                'confidence': float(face.det_score),
            })

        elapsed = time.time() - start_time
        logger.info(
            f"✅ {len(results)} face(s) in {Path(file_path).name} "
            f"({elapsed*1000:.1f}ms)"
        )
        return results

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ face_detector error ({elapsed:.3f}s): {e}")
        return []


def process_faces_from_bytes(
    image_bytes: bytes,
    is_selfie_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Same as process_faces_in_image_optimized but accepts raw image bytes.
    Useful when the image is already in memory (e.g. from an upload buffer).
    """
    start_time = time.time()
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("❌ Cannot decode image bytes")
            return []

        with _inference_lock:
            faces = face_app.get(img)

        if not faces:
            logger.warning("⚠️ No faces detected in provided bytes")
            return []

        if is_selfie_mode and len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )[:1]

        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            results.append({
                'embedding': face.embedding.tolist(),
                'bbox': {'x': x1, 'y': y1, 'w': w, 'h': h},
                'confidence': float(face.det_score),
            })

        elapsed = time.time() - start_time
        logger.info(f"✅ {len(results)} face(s) from bytes ({elapsed*1000:.1f}ms)")
        return results

    except Exception as e:
        logger.error(f"❌ face_detector bytes error: {e}")
        return []
