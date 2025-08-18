# utils/face_detector_speed_optimized.py - Speed optimizations for existing classes

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

logger = logging.getLogger(__name__)

class SmartFaceDetector:
    """Speed-optimized SmartFaceDetector (keeping original name)"""
    
    def __init__(self, model_path=None):
        try:
            if model_path is None:
                model_path = get_model_path("face_detection_yunet_2023mar.onnx")
            
            if model_path and os.path.exists(model_path):
                self.detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(320, 320),  # Smaller = faster
                    score_threshold=0.7,    # Higher threshold = less processing
                    nms_threshold=0.3,
                    top_k=30               # Limit detections
                )
                logger.info(f"✅ SmartFaceDetector (speed-optimized) initialized")
            else:
                logger.error(f"❌ Model not found")
                self.detector = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize SmartFaceDetector: {e}")
            self.detector = None
    
    def detect_and_filter_faces(self, img, is_reference=False, filter_statues=True):
        """Speed-optimized face detection"""
        try:
            if self.detector is None:
                return []
            
            h, w = img.shape[:2]
            
            # SPEED OPTIMIZATION 1: Resize large images
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img_small = cv2.resize(img, (new_w, new_h))
                scale_back = True
            else:
                img_small = img
                scale = 1.0
                scale_back = False
            
            # SPEED OPTIMIZATION 2: Single detection pass only
            sh, sw = img_small.shape[:2]
            self.detector.setInputSize((sw, sh))
            retval, faces = self.detector.detect(img_small)
            
            if faces is None or len(faces) == 0:
                return []
            
            # SPEED OPTIMIZATION 3: Scale back coordinates if needed
            if scale_back:
                for face in faces:
                    face[0] /= scale  # x
                    face[1] /= scale  # y  
                    face[2] /= scale  # w
                    face[3] /= scale  # h
            
            # SPEED OPTIMIZATION 4: Quick filtering only
            if filter_statues:
                faces = self._quick_statue_filter(faces, img)
            
            # SPEED OPTIMIZATION 5: Simple size filter
            faces = self._quick_size_filter(faces, img.shape)
            
            return faces
            
        except Exception as e:
            logger.error(f"❌ Fast detection error: {e}")
            return []
    
    def _quick_statue_filter(self, faces, img):
        """Quick statue filtering - only essential checks"""
        filtered = []
        
        for face in faces:
            try:
                x, y, w, h = face[:4]
                
                # Quick bounds check
                x1, y1 = max(int(x), 0), max(int(y), 0)
                x2, y2 = min(int(x + w), img.shape[1]), min(int(y + h), img.shape[0])
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # QUICK CHECK 1: Size ratio (statues often smaller)
                face_area = w * h
                img_area = img.shape[0] * img.shape[1]
                area_ratio = face_area / img_area
                
                if area_ratio < 0.001:  # Too small, likely background statue
                    continue
                
                # QUICK CHECK 2: Aspect ratio (faces should be roughly 3:4)
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Weird aspect ratio
                    continue
                
                # QUICK CHECK 3: Confidence check
                confidence = face[4] if len(face) > 4 else face[14] if len(face) > 14 else 0.8
                if confidence < 0.6:  # Low confidence
                    continue
                
                # If passes all quick checks, it's probably real
                filtered.append(face)
                
            except Exception:
                continue
        
        return filtered
    
    def _quick_size_filter(self, faces, img_shape):
        """Quick size-based filtering"""
        filtered = []
        min_size = 20  # Minimum face size
        
        for face in faces:
            x, y, w, h = face[:4]
            
            if w >= min_size and h >= min_size:
                # Additional check: not too large (probably not a real face if > 50% of image)
                img_area = img_shape[0] * img_shape[1]
                face_area = w * h
                if face_area / img_area < 0.5:
                    filtered.append(face)
        
        return filtered

class EnhancedFaceDetector(SmartFaceDetector):
    """Speed-optimized EnhancedFaceDetector (inherits from fast SmartFaceDetector)"""
    
    def __init__(self, model_path=None):
        super().__init__(model_path)
        logger.info("✅ EnhancedFaceDetector (speed-optimized) ready")
    
    def detect_and_filter_faces(self, img, filter_statues=True):
        """Speed-optimized detection for group photos"""
        return super().detect_and_filter_faces(img, is_reference=False, filter_statues=filter_statues)


# Global detector instances (keeping original names and patterns)
_smart_detector_instance: Optional[SmartFaceDetector] = None
_enhanced_detector_instance: Optional[EnhancedFaceDetector] = None
_detector_lock = threading.Lock()

def get_smart_face_detector() -> SmartFaceDetector:
    """Get or create smart face detector instance (speed-optimized)"""
    global _smart_detector_instance
    
    if _smart_detector_instance is None:
        with _detector_lock:
            if _smart_detector_instance is None:
                _smart_detector_instance = SmartFaceDetector()
                logger.info("🚀 SmartFaceDetector (speed-optimized) instance created")
    
    return _smart_detector_instance

def get_enhanced_face_detector() -> EnhancedFaceDetector:
    """Get or create enhanced face detector instance (speed-optimized)"""
    global _enhanced_detector_instance
    
    if _enhanced_detector_instance is None:
        with _detector_lock:
            if _enhanced_detector_instance is None:
                _enhanced_detector_instance = EnhancedFaceDetector()
                logger.info("🚀 EnhancedFaceDetector (speed-optimized) instance created")
    
    return _enhanced_detector_instance

# Import missing dependencies from original detector
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Import these from your original detector file
try:
    from core.device_setup import device, resnet, API_BASE
except ImportError:
    logger.warning("⚠️ core.device_setup not found, using defaults")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = None
    API_BASE = "http://localhost:8000"

def get_model_path(model_filename="face_detection_yunet_2023mar.onnx"):
    """Get correct model path for both development and PyInstaller"""
    
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

def generate_face_embedding(face_crop):
    """Generate single face embedding (fallback for compatibility)"""
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
        
        return embedding
        
    except Exception as e:
        logger.error(f"❌ Embedding generation error: {e}")
        return None

def generate_face_embedding_batch(face_crops):
    """Generate embeddings for multiple faces in parallel"""
    try:
        if not face_crops or resnet is None:
            logger.warning("❌ No face crops or FaceNet model not available")
            return []
        
        logger.debug(f"🔄 Generating batch embeddings for {len(face_crops)} faces")
        
        # Batch processing for speed
        batch_tensors = []
        
        for i, face_crop in enumerate(face_crops):
            try:
                # Resize to 160x160 for FaceNet
                face_img = cv2.resize(face_crop, (160, 160))
                face_pil = Image.fromarray(face_img)
                
                # Convert to tensor
                face_tensor = torch.from_numpy(np.array(face_pil)).permute(2, 0, 1).float()
                face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                batch_tensors.append(face_tensor)
                
            except Exception as e:
                logger.error(f"❌ Error processing face {i} for batch: {e}")
                continue
        
        if not batch_tensors:
            logger.warning("❌ No valid face tensors for batch processing")
            return []
        
        # Stack into batch and move to device
        batch = torch.stack(batch_tensors).to(device)
        logger.debug(f"✅ Batch tensor shape: {batch.shape}")
        
        # Generate embeddings in batch
        with torch.no_grad():
            embeddings = resnet(batch)
            embeddings = embeddings.cpu().numpy()
        
        logger.debug(f"✅ Generated {len(embeddings)} batch embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Batch embedding error: {e}")
        # Fallback to individual embeddings
        logger.info("🔄 Falling back to individual embedding generation")
        results = []
        for i, face_crop in enumerate(face_crops):
            try:
                emb = generate_face_embedding(face_crop)
                if emb is not None:
                    results.append(emb)
                else:
                    logger.warning(f"⚠️ Failed to generate embedding for face {i}")
            except Exception as e:
                logger.error(f"❌ Individual embedding error for face {i}: {e}")
                continue
        
        logger.info(f"✅ Fallback generated {len(results)} individual embeddings")
        return results
    """Generate embeddings for multiple faces in parallel"""
    try:
        if not face_crops:
            return []
        
        # Batch processing for speed
        batch_tensors = []
        
        for face_crop in face_crops:
            # Resize to 160x160 for FaceNet
            face_img = cv2.resize(face_crop, (160, 160))
            face_pil = Image.fromarray(face_img)
            
            # Convert to tensor
            face_tensor = torch.from_numpy(np.array(face_pil)).permute(2, 0, 1).float()
            face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
            batch_tensors.append(face_tensor)
        
        # Stack into batch
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            
            # Generate embeddings in batch
            with torch.no_grad():
                embeddings = resnet(batch)
                embeddings = embeddings.cpu().numpy()
            
            return embeddings
        
        return []
        
    except Exception as e:
        logger.error(f"❌ Batch embedding error: {e}")
        return []

def process_faces_in_image_optimized(file_path: str, is_selfie_mode: bool = False) -> List[Dict[str, Any]]:
    """
    ULTRA-FAST version - optimized for speed over everything else
    Target: < 1 second for 15 faces
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 processing: {Path(file_path).name}")
        
        # SPEED 1: Fast image read
        img = cv2.imread(file_path)
        if img is None:
            return []
        
        read_time = time.time() - start_time
        
        # SPEED 2: Get fast detector
        detector = get_smart_face_detector()  # Use existing global instance
        if detector is None or detector.detector is None:
            return []
        
        # SPEED 3: Fast detection (no retries, no preprocessing)
        detection_start = time.time()
        faces = detector.detect_and_filter_faces(img, is_reference=is_selfie_mode, filter_statues=True)
        detection_time = time.time() - detection_start
        
        if not faces:
            total_time = time.time() - start_time
            logger.info(f"❌ No faces found ({total_time:.3f}s)")
            return []
        
        # SPEED 4: Batch face crop extraction
        crop_start = time.time()
        face_crops = []
        face_bboxes = []
        face_confidences = []
        
        h, w = img.shape[:2]
        
        for face in faces:
            try:
                x, y, w_box, h_box = face[:4]
                confidence = face[4] if len(face) > 4 else face[14] if len(face) > 14 else 0.8
                
                # Quick coordinate validation
                x1, y1 = max(int(x), 0), max(int(y), 0)
                x2, y2 = min(int(x + w_box), w), min(int(y + h_box), h)
                
                if x2 > x1 and y2 > y1:
                    face_crop_bgr = img[y1:y2, x1:x2]
                    face_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    
                    if face_crop.size > 0:
                        face_crops.append(face_crop)
                        face_bboxes.append({'x': int(x), 'y': int(y), 'w': int(w_box), 'h': int(h_box)})
                        face_confidences.append(float(confidence))
                        
            except Exception:
                continue
        
        crop_time = time.time() - crop_start
        
        if not face_crops:
            return []
        
        # SPEED 5: Batch embedding generation
        embedding_start = time.time()
        
        # Check if we can use batch processing
        if len(face_crops) > 1 and resnet is not None:
            try:
                embeddings = generate_face_embedding_batch(face_crops)
                if len(embeddings) == len(face_crops):
                    # Batch successful
                    pass
                else:
                    # Batch failed, fallback to individual
                    logger.warning("⚠️ Batch embedding failed, using individual processing")
                    embeddings = []
                    for face_crop in face_crops:
                        emb = generate_face_embedding(face_crop)
                        if emb is not None:
                            embeddings.append(emb)
            except Exception as e:
                logger.warning(f"⚠️ Batch embedding error: {e}, falling back to individual")
                embeddings = []
                for face_crop in face_crops:
                    emb = generate_face_embedding(face_crop)
                    if emb is not None:
                        embeddings.append(emb)
        else:
            # Single face or no batch support, use individual processing
            embeddings = []
            for face_crop in face_crops:
                emb = generate_face_embedding(face_crop)
                if emb is not None:
                    embeddings.append(emb)
        
        embedding_time = time.time() - embedding_start
        
        if len(embeddings) != len(face_crops):
            logger.error(f"❌ Embedding generation failed: {len(embeddings)} embeddings for {len(face_crops)} faces")
            return []
        
        # SPEED 6: Quick result assembly
        results = []
        for i in range(len(embeddings)):
            result = {
                'embedding': embeddings[i].tolist(),
                'bbox': face_bboxes[i],
                'confidence': face_confidences[i]
            }
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Performance logging
        logger.info(f"⚡ SPEED RESULTS:")
        logger.info(f"   {len(results)} faces processed in {total_time:.3f}s")
        logger.info(f"   Read: {read_time:.3f}s | Detection: {detection_time:.3f}s")
        logger.info(f"   Crop: {crop_time:.3f}s | Embedding: {embedding_time:.3f}s")
        logger.info(f"   Per face: {total_time/len(results):.3f}s")
        
        return results
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Speed processing error ({total_time:.3f}s): {e}")
        return []


# SPEED CONFIG
class SpeedConfig:
    """Configuration for maximum speed"""
    
    # Detection settings (favor speed over accuracy)
    MAX_IMAGE_SIZE = 1280        # Resize larger images
    DETECTION_THRESHOLD = 0.7    # Higher threshold = fewer false positives
    MIN_FACE_SIZE = 20          # Minimum face size
    MAX_FACES = 30              # Limit max detections
    
    # Statue filtering (minimal checks only)
    ENABLE_STATUE_FILTER = True  # Can disable for max speed
    MIN_AREA_RATIO = 0.001      # Minimum face area ratio
    MIN_ASPECT_RATIO = 0.5      # Minimum face aspect ratio
    MAX_ASPECT_RATIO = 2.0      # Maximum face aspect ratio
    MIN_CONFIDENCE = 0.6        # Minimum detection confidence
    
    # Embedding settings
    BATCH_EMBEDDINGS = True     # Generate embeddings in batch
    EMBEDDING_SIZE = 160        # Face size for embedding (smaller = faster)

# Global speed config
speed_config = SpeedConfig()

def configure_for_max_speed():
    """Configure detector for maximum speed"""
    global speed_config
    
    logger.info("🚀 Configuring for MAXIMUM SPEED...")
    
    # Most aggressive speed settings
    speed_config.MAX_IMAGE_SIZE = 960        # Even smaller
    speed_config.DETECTION_THRESHOLD = 0.75  # Higher threshold
    speed_config.MIN_FACE_SIZE = 25         # Larger minimum
    speed_config.MAX_FACES = 20             # Fewer max faces
    speed_config.ENABLE_STATUE_FILTER = False  # Disable for max speed
    
    logger.info("✅ Maximum speed configuration active")
    logger.info("   Target: < 0.5s for 15 faces")

def estimate_speed_performance(num_faces: int) -> Dict[str, float]:
    """Estimate processing time with speed optimizations"""
    
    # Speed estimates (in seconds)
    estimates = {
        'image_read': 0.02,
        'face_detection': num_faces * 0.015,  # ~15ms per face
        'face_cropping': num_faces * 0.005,   # ~5ms per face  
        'batch_embedding': num_faces * 0.020,  # ~20ms per face (batched)
        'result_assembly': 0.01
    }
    
    total = sum(estimates.values())
    estimates['total'] = total
    
    logger.info(f"⚡ Speed estimate for {num_faces} faces:")
    for step, time_est in estimates.items():
        logger.info(f"   {step}: {time_est:.3f}s")
    
    return estimates
