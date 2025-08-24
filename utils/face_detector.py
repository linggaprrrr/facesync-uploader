# utils/face_detector_speed_optimized.py - Adapted from your working app

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import logging
import time
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import sys
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global face detector instance (like your working app)
face_detector = None

def init_face_detector():
    """Initialize face detector exactly like your working app"""
    global face_detector
    if face_detector is None:
        model_path = get_model_path("face_detection_yunet_2023mar.onnx")
        if model_path:
            face_detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(640, 640),
                score_threshold=0.6,    # Same as your working app
                nms_threshold=0.3,      # Same as your working app
                top_k=120             # Same as your working app
            )
            
        if face_detector is None:
            logger.error("❌ Gagal inisialisasi FaceDetectorYN.")
        else:
            logger.info("✅ FaceDetectorYN berhasil diinisialisasi.")

def letterbox_resize(img, target_size=(640, 640)):
    """Letterbox resize function (adapted from your working app)"""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # Gray padding
    
    # Center the resized image
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    pad = (start_x, start_y)
    
    return padded, scale, pad

def reverse_letterbox(bbox, original_shape, resized_shape, pad, scale):
    """Reverse letterbox transformation (from your working app)"""
    try:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        # Remove padding
        x = x - pad[0]
        y = y - pad[1]
        
        # Scale back to original
        x = x / scale
        y = y / scale
        w = w / scale
        h = h / scale
        
        # Ensure bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, original_shape[1] - x)
        h = min(h, original_shape[0] - y)
        
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    except Exception as e:
        logger.error(f"❌ Reverse letterbox error: {e}")
        return bbox

class SmartFaceDetector:
    """Wrapper using your working app's pattern"""
    
    def __init__(self, model_path=None):
        # Initialize the global detector
        init_face_detector()
        logger.info("✅ SmartFaceDetector (working app pattern) initialized")
    
    def detect_and_filter_faces(self, img, is_reference=False, filter_statues=True):
        """Detection using your working app's exact pattern"""
        try:
            global face_detector
            
            if face_detector is None:
                logger.error("❌ Face detector not initialized")
                return []
            
            # Get original shape
            original_shape = img.shape[:2]
            
            # Letterbox resize (exactly like your working app)
            resized_img, scale, pad = letterbox_resize(img, (640, 640))
            
            # Set input size and detect (exactly like your working app)
            face_detector.setInputSize((resized_img.shape[1], resized_img.shape[0]))
            retval, faces = face_detector.detect(resized_img)
            
            logger.info(f"🔍 Raw detection: {0 if faces is None else len(faces)} faces")
            
            if faces is None or len(faces) == 0:
                return []
            
            # Process faces (adapted from your working app)
            processed_faces = []
            
            for i, face in enumerate(faces):
                try:
                    x, y, w, h = map(int, face[:4])
                    confidence = face[4] if len(face) > 4 else face[14] if len(face) > 14 else 0.8
                    
                    # Basic validation
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Create bbox dict
                    bbox_dict = {"x": x, "y": y, "w": w, "h": h}
                    
                    # Reverse letterbox to get original coordinates
                    original_bbox = reverse_letterbox(
                        bbox=bbox_dict,
                        original_shape=original_shape,
                        resized_shape=resized_img.shape[:2],
                        pad=pad,
                        scale=scale
                    )
                    
                    # Convert back to face format for compatibility
                    processed_face = [
                        float(original_bbox["x"]),
                        float(original_bbox["y"]),
                        float(original_bbox["w"]),
                        float(original_bbox["h"]),
                        float(confidence)
                    ]
                    
                    # Add remaining values if they exist
                    if len(face) > 5:
                        processed_face.extend(face[5:])
                    
                    processed_faces.append(processed_face)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing face {i}: {e}")
                    continue
            
            # Reference mode: select largest face (like your working app)
            if is_reference and processed_faces:
                processed_faces = sorted(processed_faces, key=lambda f: f[2] * f[3], reverse=True)
                processed_faces = [processed_faces[0]]  # Only the largest
            
            logger.info(f"✅ Processed faces: {len(processed_faces)}")
            return processed_faces
            
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            return []

class EnhancedFaceDetector(SmartFaceDetector):
    """Enhanced detector using working app pattern"""
    
    def __init__(self, model_path=None):
        super().__init__(model_path)
        logger.info("✅ EnhancedFaceDetector (working app pattern) ready")
    
    def detect_and_filter_faces(self, img, filter_statues=True):
        """Detection for group photos"""
        return super().detect_and_filter_faces(img, is_reference=False, filter_statues=filter_statues)

# Global detector instances
_smart_detector_instance: Optional[SmartFaceDetector] = None
_enhanced_detector_instance: Optional[EnhancedFaceDetector] = None
_detector_lock = threading.Lock()

def get_smart_face_detector() -> SmartFaceDetector:
    """Get smart face detector instance"""
    global _smart_detector_instance
    
    if _smart_detector_instance is None:
        with _detector_lock:
            if _smart_detector_instance is None:
                _smart_detector_instance = SmartFaceDetector()
                logger.info("🎯 SmartFaceDetector (working app pattern) instance created")
    
    return _smart_detector_instance

def get_enhanced_face_detector() -> EnhancedFaceDetector:
    """Get enhanced face detector instance"""
    global _enhanced_detector_instance
    
    if _enhanced_detector_instance is None:
        with _detector_lock:
            if _enhanced_detector_instance is None:
                _enhanced_detector_instance = EnhancedFaceDetector()
                logger.info("🎯 EnhancedFaceDetector (working app pattern) instance created")
    
    return _enhanced_detector_instance

# Import existing core modules
try:
    from core.device_setup import device, resnet, API_BASE
except ImportError:
    logger.warning("⚠️ core.device_setup not found, using defaults")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = None
    API_BASE = os.getenv('API_BASE', 'http://localhost:8001')

def get_model_path(model_filename="face_detection_yunet_2023mar.onnx"):
    """Get correct model path"""
    
    # If running in PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        bundle_model_path = os.path.join(sys._MEIPASS, 'models', model_filename)
        if os.path.exists(bundle_model_path):
            return bundle_model_path
    
    # Development mode paths
    dev_paths = [
        f"models/{model_filename}",
        model_filename,
        f"./{model_filename}",
        os.path.join(os.getcwd(), "models", model_filename)
    ]
    
    for path in dev_paths:
        if os.path.exists(path):
            return path
    
    logger.error(f"❌ Model {model_filename} not found!")
    return None

def preprocess_face_batch(face_crops):
    """OPTIMIZED: Preprocess faces for embedding"""
    try:
        if not face_crops:
            return None
            
        # Pre-allocate numpy array for better performance
        batch_size = len(face_crops)
        batch_array = np.zeros((batch_size, 3, 160, 160), dtype=np.float32)
        
        for i, face_crop in enumerate(face_crops):
            # Resize to 160x160 for FaceNet
            face_img = cv2.resize(face_crop, (160, 160))
            # Convert BGR to RGB and normalize in one step
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_normalized = (face_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            # Transpose to CHW format
            batch_array[i] = face_normalized.transpose(2, 0, 1)
        
        # Convert to tensor and move to device in one operation
        batch_tensor = torch.from_numpy(batch_array).to(device, non_blocking=True)
        return batch_tensor
        
    except Exception as e:
        logger.error(f"❌ Preprocess batch error: {e}")
        return None

def l2_normalize(embedding):
    """L2 normalize embedding (from your working app)"""
    try:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    except Exception as e:
        logger.error(f"❌ L2 normalize error: {e}")
        return embedding

def generate_face_embedding(face_crop):
    """Generate single face embedding"""
    try:
        if resnet is None:
            logger.error("❌ FaceNet model not available")
            return None
            
        # Resize to 160x160 for FaceNet
        face_img = cv2.resize(face_crop, (160, 160))
        face_pil = Image.fromarray(face_img)
        
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(np.array(face_pil)).permute(2, 0, 1).float()
        face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
        
        # Add batch dimension and move to device
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = resnet(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
            embedding = l2_normalize(embedding)  # Apply L2 normalization
        
        return embedding
        
    except Exception as e:
        logger.error(f"❌ Embedding generation error: {e}")
        return None

def generate_face_embedding_batch(face_crops):
    """OPTIMIZED: Generate batch embeddings with performance improvements"""
    try:
        if not face_crops or resnet is None:
            logger.warning("No face crops or FaceNet model not available")
            return []
        
        logger.info(f"Generating batch embeddings for {len(face_crops)} faces")
        
        # Optimized preprocessing
        face_tensors = preprocess_face_batch(face_crops)
        
        if face_tensors is None:
            logger.error("Failed to preprocess faces")
            return []
        
        # Generate embeddings with optimizations
        start_inference = time.time()
        with torch.no_grad():
            # Ensure model is in eval mode
            resnet.eval()
            # Use torch.inference_mode for better performance
            with torch.inference_mode():
                embeddings_tensor = resnet(face_tensors)
        
        inference_time = time.time() - start_inference
        logger.info(f"Model inference time: {inference_time:.3f}s ({inference_time/len(face_crops)*1000:.1f}ms per face)")
        
        # Convert to numpy and normalize efficiently
        embeddings = embeddings_tensor.cpu().numpy()
        
        # Vectorized L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        logger.info(f"Generated {len(embeddings)} batch embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        # Fallback to individual embeddings
        results = []
        for face_crop in face_crops:
            emb = generate_face_embedding(face_crop)
            if emb is not None:
                results.append(emb)
        return results

def process_faces_in_image_optimized(file_path: str, is_selfie_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Process faces using your working app's exact pattern
    """
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Processing (working app pattern): {Path(file_path).name}")
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"❌ Could not load image: {file_path}")
            return []
        
        logger.info(f"🖼️ Image loaded successfully. Shape: {img.shape}")
        
        # Get detector
        detector = get_smart_face_detector()
        
        # Detect faces using working app pattern
        detection_start = time.time()
        faces = detector.detect_and_filter_faces(img, is_reference=is_selfie_mode, filter_statues=True)
        detection_time = time.time() - detection_start
        
        if not faces:
            total_time = time.time() - start_time
            logger.warning(f"❌ No faces found ({total_time:.3f}s)")
            return []
        
        logger.info(f"✅ Found {len(faces)} faces")
        
        # Extract face crops (using original image coordinates)
        face_crops = []
        face_bboxes = []
        face_confidences = []
        
        h, w = img.shape[:2]
        
        for i, face in enumerate(faces):
            try:
                x, y, face_w, face_h = face[:4]
                confidence = face[4] if len(face) > 4 else 0.8
                
                x, y, face_w, face_h = int(x), int(y), int(face_w), int(face_h)
                
                # Ensure bounds
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + face_w, w), min(y + face_h, h)
                
                if x2 > x1 and y2 > y1:
                    face_crop = img[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        logger.info(f"✅ Face {i} cropped successfully. Shape: {face_crop.shape}")
                        face_crops.append(face_crop)
                        face_bboxes.append({'x': x, 'y': y, 'w': face_w, 'h': face_h})
                        face_confidences.append(float(confidence))
                    else:
                        logger.warning(f"⚠️ Face {i} crop empty. Skip.")
                        
            except Exception as e:
                logger.error(f"❌ Face crop error for face {i}: {e}")
                continue
        
        if not face_crops:
            logger.error("❌ No valid face crops extracted")
            return []
        
        logger.info(f"🧼 Preprocessing {len(face_crops)} faces for embedding")
        
        # Generate embeddings using working app pattern
        embedding_start = time.time()
        embeddings = generate_face_embedding_batch(face_crops)
        embedding_time = time.time() - embedding_start
        
        if len(embeddings) != len(face_crops):
            logger.error(f"❌ Embedding count mismatch: {len(embeddings)} vs {len(face_crops)}")
            return []
        
        # Assemble results
        results = []
        for i in range(len(embeddings)):
            result = {
                'embedding': embeddings[i].tolist(),
                'bbox': face_bboxes[i],
                'confidence': face_confidences[i]
            }
            results.append(result)
        
        total_time = time.time() - start_time
        
        logger.info(f"🎯 WORKING APP PATTERN RESULTS:")
        logger.info(f"   {len(results)} faces processed in {total_time:.3f}s")
        logger.info(f"   Detection: {detection_time:.3f}s | Embedding: {embedding_time:.3f}s")
        logger.info(f"   Per face: {total_time/len(results):.3f}s")
        
        return results
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Processing error ({total_time:.3f}s): {e}")
        return []

# Initialize detector on module import
init_face_detector()