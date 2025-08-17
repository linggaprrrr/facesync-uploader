# utils/face_detector.py - Updated with missing function

import asyncio
import aiohttp
import json
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from uuid import UUID
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from dotenv import load_dotenv
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)
load_dotenv()
from core.device_setup import device, resnet, API_BASE

@dataclass
class UploadConfig:
    """Configuration for optimized uploads"""
    api_base_url: str = API_BASE
    max_concurrent_data: int = 5  # Concurrent data uploads
    max_concurrent_files: int = 3  # Concurrent file uploads
    batch_size_data: int = 20  # Photos per data batch
    batch_size_files: int = 10  # Files per file batch
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout_data: int = 30
    timeout_files: int = 120
    status_check_interval: float = 2.0
    max_status_checks: int = 30

@dataclass
class PhotoUploadItem:
    """Single photo upload item"""
    file_path: str
    unit_id: str
    outlet_id: str
    photo_type_id: str
    faces_data: List[Dict[str, Any]]
    filename: str = None
    
    def __post_init__(self):
        if self.filename is None:
            self.filename = Path(self.file_path).name

@dataclass
class UploadResult:
    """Upload result tracking"""
    success: bool
    photo_id: Optional[UUID] = None
    original_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0

# ===== MISSING FUNCTION: Face Processing =====

# ===== MISSING FUNCTION: Advanced Face Processing with SmartFaceDetector =====

import math
from PIL import Image, ExifTags

def get_model_path(model_filename="face_detection_yunet_2023mar.onnx"):
    """Get correct model path for both development and PyInstaller"""
    
    # If running in PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller bundle path
        bundle_model_path = os.path.join(sys._MEIPASS, 'models', model_filename)
        print(f"🔍 Checking PyInstaller bundle path: {bundle_model_path}")
        
        if os.path.exists(bundle_model_path):
            print(f"✅ Found model in bundle: {bundle_model_path}")
            return bundle_model_path
        else:
            print(f"❌ Model not found in bundle: {bundle_model_path}")
            # List what's actually in the bundle models folder
            bundle_models_dir = os.path.join(sys._MEIPASS, 'models')
            if os.path.exists(bundle_models_dir):
                files = os.listdir(bundle_models_dir)
                print(f"📁 Bundle models folder contains: {files}")
            else:
                print(f"❌ Bundle models folder doesn't exist: {bundle_models_dir}")
                # List bundle root contents
                bundle_contents = os.listdir(sys._MEIPASS)
                print(f"📁 Bundle root contains: {bundle_contents[:10]}...")
    
    # Development mode paths
    dev_paths = [
        f"models/{model_filename}",
        model_filename,
        f"./{model_filename}",
        os.path.join(os.getcwd(), "models", model_filename)
    ]
    
    print("🔍 Checking development paths:")
    for path in dev_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            print(f"✅ Found model in development: {path}")
            return path
        else:
            print(f"❌ Not found: {path}")
    
    print(f"❌ Model {model_filename} not found in any location!")
    return None

class SmartFaceDetector:
    """Smart face detector untuk registration dengan auto-rotate dan face filtering"""
    
    def __init__(self, model_path=None):
        try:
            # Use the new path detection function
            if model_path is None:
                model_path = get_model_path("face_detection_yunet_2023mar.onnx")
            
            if model_path and os.path.exists(model_path):
                self.detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(640, 640),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=100
                )
                logger.info(f"✅ SmartFaceDetector initialized with: {model_path}")
            else:
                logger.error(f"❌ Model not found, tried: {model_path}")
                self.detector = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize SmartFaceDetector: {e}")
            self.detector = None
    
    def fix_image_orientation(self, image_data, force_portrait=True):
        """
        Rotate image using EXIF and shape detection
        
        Args:
            image_data: Raw image bytes
            force_portrait: If True, ensure output is portrait orientation
        """
        try:
            pil_image = Image.open(BytesIO(image_data))

            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Step 1: Apply EXIF rotation first
            exif = pil_image.getexif()
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)

            if orientation_key and orientation_key in exif:
                orientation = exif.get(orientation_key)
                if orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                    logger.info("🔄 Rotated 180° via EXIF")
                elif orientation == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                    logger.info("🔄 Rotated 270° via EXIF")
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
                    logger.info("🔄 Rotated 90° via EXIF")
                else:
                    logger.info("ℹ️ No EXIF rotation needed (orientation = 1)")
            else:
                logger.info("ℹ️ No EXIF info found")

            # Step 2: Check if we need shape-based rotation
            width, height = pil_image.size
            is_landscape = width > height
            
            if force_portrait and is_landscape:
                # Detect faces to determine correct rotation
                temp_buffer = BytesIO()
                pil_image.save(temp_buffer, format="JPEG", quality=95)
                temp_data = temp_buffer.getvalue()
                
                img_array = np.frombuffer(temp_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Try current orientation
                faces_current = self.detect_faces_simple(img)
                
                # Try 90° rotation
                img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                faces_90 = self.detect_faces_simple(img_90)
                
                # Try 270° rotation
                img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                faces_270 = self.detect_faces_simple(img_270)
                
                # Compare face detection quality
                score_current = self.evaluate_face_detection(faces_current, img.shape)
                score_90 = self.evaluate_face_detection(faces_90, img_90.shape)
                score_270 = self.evaluate_face_detection(faces_270, img_270.shape)
                
                logger.info(f"Face detection scores - Current: {score_current:.3f}, "
                           f"90°: {score_90:.3f}, 270°: {score_270:.3f}")
                
                # Choose best rotation
                if score_90 > max(score_current, score_270) and score_90 > 0:
                    pil_image = pil_image.rotate(270, expand=True)  # PIL rotate is counter-clockwise
                    logger.info("🔄 Auto-rotated 90° clockwise for portrait")
                elif score_270 > max(score_current, score_90) and score_270 > 0:
                    pil_image = pil_image.rotate(90, expand=True)   # PIL rotate is counter-clockwise
                    logger.info("🔄 Auto-rotated 270° clockwise for portrait")
                else:
                    # If no clear winner, default to 90° rotation for portrait
                    if is_landscape:
                        pil_image = pil_image.rotate(270, expand=True)
                        logger.info("🔄 Default 90° rotation for portrait (no clear face direction)")

            # Final check
            final_width, final_height = pil_image.size
            logger.info(f"📐 Final dimensions: {final_width}x{final_height} "
                       f"({'Portrait' if final_height > final_width else 'Landscape'})")

            # Encode final result
            output_buffer = BytesIO()
            pil_image.save(output_buffer, format="JPEG", quality=95)
            corrected_data = output_buffer.getvalue()

            img_array = np.frombuffer(corrected_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return img, corrected_data

        except Exception as e:
            logger.warning(f"⚠️ Failed to fix orientation: {e}")
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img, image_data

    def detect_faces_simple(self, img):
        """Simple face detection for orientation checking"""
        try:
            if self.detector is None:
                return []
            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            retval, faces = self.detector.detect(img)
            return faces if faces is not None else []
        except:
            return []
    
    def evaluate_face_detection(self, faces, img_shape):
        """Evaluate face detection quality for orientation decision"""
        if faces is None or len(faces) == 0:
            return 0.0
        
        total_score = 0.0
        for face in faces:
            metrics = self.calculate_face_metrics(face, img_shape)
            # Prioritize larger, centered faces with good confidence
            face_score = (
                metrics['area_ratio'] * 0.5 +  # Size matters most
                (1 - metrics['center_distance']) * 0.3 +  # Center position
                metrics['confidence'] * 0.2  # Detection confidence
            )
            total_score += face_score
        
        # Bonus for detecting exactly one face (typical for selfie)
        if len(faces) == 1:
            total_score *= 1.2
        
        return total_score
    
    def calculate_face_metrics(self, face, img_shape):
        """Calculate face metrics untuk filtering"""
        x, y, w, h = face[:4]
        img_h, img_w = img_shape[:2]
        
        # Face area
        face_area = w * h
        image_area = img_w * img_h
        area_ratio = face_area / image_area
        
        # Center distance (dari center image)
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        center_distance = math.sqrt(
            (face_center_x - img_center_x) ** 2 + 
            (face_center_y - img_center_y) ** 2
        )
        
        # Normalize center distance (0-1)
        max_distance = math.sqrt(img_center_x ** 2 + img_center_y ** 2)
        normalized_center_distance = center_distance / max_distance if max_distance > 0 else 0
        
        # Aspect ratio (ideal face ratio ≈ 0.75)
        face_aspect_ratio = h / w if w > 0 else 0
        aspect_score = 1 - abs(face_aspect_ratio - 0.75) / 0.75
        
        return {
            'area_ratio': area_ratio,
            'center_distance': normalized_center_distance,
            'aspect_score': max(0, aspect_score),
            'confidence': float(face[14]) if len(face) > 14 else float(face[4])
        }
    
    def is_likely_selfie_face(self, face, img_shape, min_area_ratio=0.05):
        """Determine if face is likely the main selfie subject"""
        metrics = self.calculate_face_metrics(face, img_shape)
        
        # Kriteria selfie face:
        # 1. Area cukup besar (minimal 5% dari image)
        # 2. Posisi relatif di tengah
        # 3. Confidence tinggi
        
        area_ok = metrics['area_ratio'] >= min_area_ratio
        center_ok = metrics['center_distance'] <= 0.6  # Not too far from center
        confidence_ok = metrics['confidence'] >= 0.7
        
        # Composite score
        selfie_score = (
            metrics['area_ratio'] * 0.4 +  # 40% weight on size
            (1 - metrics['center_distance']) * 0.3 +  # 30% weight on center position
            metrics['confidence'] * 0.2 +  # 20% weight on confidence
            metrics['aspect_score'] * 0.1   # 10% weight on aspect ratio
        )
        
        logger.debug(f"Face metrics: area={metrics['area_ratio']:.3f}, "
                    f"center_dist={metrics['center_distance']:.3f}, "
                    f"conf={metrics['confidence']:.3f}, "
                    f"selfie_score={selfie_score:.3f}")
        
        return area_ok and center_ok and confidence_ok, selfie_score
    
    def detect_and_filter_faces(self, img, is_reference=True):
        """Detect faces dengan smart filtering untuk selfie/reference"""
        try:
            if self.detector is None:
                logger.error("❌ Detector not initialized")
                return []
                
            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            
            retval, faces = self.detector.detect(img)
            
            if faces is None or len(faces) == 0:
                logger.warning("❌ No faces detected")
                return []
            
            logger.info(f"🔍 Detected {len(faces)} faces")
            
            if not is_reference:
                # For non-reference, return all faces
                return faces
            
            # For reference registration, filter untuk selfie
            selfie_faces = []
            face_scores = []
            
            for i, face in enumerate(faces):
                is_selfie, score = self.is_likely_selfie_face(face, img.shape)
                metrics = self.calculate_face_metrics(face, img.shape)
                
                logger.info(f"Face {i}: area_ratio={metrics['area_ratio']:.3f}, "
                           f"confidence={metrics['confidence']:.3f}, "
                           f"selfie_score={score:.3f}, is_selfie={is_selfie}")
                
                if is_selfie:
                    selfie_faces.append(face)
                    face_scores.append(score)
            
            if not selfie_faces:
                # Fallback: ambil face dengan area terbesar jika tidak ada yang memenuhi kriteria selfie
                logger.warning("⚠️ No selfie faces found, using largest face")
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                return [largest_face]
            
            # Sort by selfie score dan ambil yang terbaik
            sorted_faces = [face for _, face in sorted(zip(face_scores, selfie_faces), 
                                                     key=lambda x: x[0], reverse=True)]
            
            # Untuk reference, hanya ambil 1 face terbaik
            best_face = sorted_faces[0]
            best_score = max(face_scores)
            
            logger.info(f"✅ Selected best selfie face with score: {best_score:.3f}")
            return [best_face]
            
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
            return []

# Global smart detector instance
_smart_detector_instance: Optional[SmartFaceDetector] = None
_smart_detector_lock = threading.Lock()

def get_smart_face_detector() -> SmartFaceDetector:
    """Get or create smart face detector instance"""
    global _smart_detector_instance
    
    if _smart_detector_instance is None:
        with _smart_detector_lock:
            if _smart_detector_instance is None:
                _smart_detector_instance = SmartFaceDetector()
                logger.info("🚀 SmartFaceDetector instance created")
    
    return _smart_detector_instance

def process_faces_in_image_optimized(file_path: str, is_selfie_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Process faces in image using SmartFaceDetector with configurable detection mode
    
    Args:
        file_path: Path to image file
        is_selfie_mode: If True, applies selfie filtering. If False, detects all faces.
        
    Returns:
        List of dictionaries containing:
        - embedding: List of floats (512-dim face embedding)
        - bbox: Dict with x, y, w, h coordinates
        - confidence: Float confidence score
    """
    try:
        logger.info(f"🔍 Starting face processing ({'selfie mode' if is_selfie_mode else 'all faces mode'}): {Path(file_path).name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return []
        
        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"📄 File size: {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"❌ Empty file: {file_path}")
            return []
        
        # Read image as bytes for orientation fixing
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()
            logger.info(f"✅ Successfully read {len(image_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            return []
        
        # Get smart detector
        smart_detector = get_smart_face_detector()
        if smart_detector is None or smart_detector.detector is None:
            logger.error("❌ SmartFaceDetector not available - trying fallback")
            return _fallback_face_detection(file_path)
        
        logger.info("✅ SmartFaceDetector is ready")
        
        # Fix image orientation only for selfie mode, skip for ride photos
        if is_selfie_mode:
            try:
                img, corrected_data = smart_detector.fix_image_orientation(image_data, force_portrait=True)
                logger.info(f"✅ Orientation fixed, image shape: {img.shape if img is not None else 'None'}")
            except Exception as e:
                logger.error(f"❌ Orientation fixing failed: {e}")
                # Fallback: try to read image directly
                img = cv2.imread(file_path)
                if img is not None:
                    logger.warning("⚠️ Using direct image read as fallback")
                else:
                    logger.error(f"❌ Complete image read failure: {file_path}")
                    return []
        else:
            # For ride photos, read directly without orientation fixing
            img = cv2.imread(file_path)
            if img is None:
                logger.error(f"❌ Failed to read image: {file_path}")
                return []
            logger.info("✅ Image read directly (skipping orientation fix for ride photos)")
        
        if img is None:
            logger.error(f"❌ Failed to read/process image: {file_path}")
            return []
        
        h, w, c = img.shape
        logger.info(f"📐 Final image dimensions: {w}x{h}x{c}")
        
        # Detect faces with mode-specific filtering
        faces = None
        
        if is_selfie_mode:
            # Use smart filtering for selfies
            logger.info("🎯 Using selfie-optimized detection")
            faces = smart_detector.detect_and_filter_faces(img, is_reference=True)
        else:
            # Detect ALL faces for ride photos (no selfie filtering)
            logger.info("🎠 Using all-faces detection for ride photos")
            faces = smart_detector.detect_and_filter_faces(img, is_reference=False)
        
        # Fix: Check faces properly (could be numpy array or list)
        if faces is None or len(faces) == 0:
            detection_attempts = [
                {"threshold": 0.5, "description": "Lower threshold"}, 
                {"threshold": 0.4, "description": "Very low threshold"},
                {"threshold": 0.3, "description": "Minimal threshold"},
            ]
            
            for attempt in detection_attempts:
                try:
                    logger.info(f"🔍 Retry with {attempt['description']} (threshold: {attempt['threshold']})")
                    
                    # Temporarily adjust detector threshold
                    original_threshold = smart_detector.detector.getScoreThreshold()
                    smart_detector.detector.setScoreThreshold(attempt['threshold'])
                    
                    # Try raw detection without any filtering
                    h, w = img.shape[:2]
                    smart_detector.detector.setInputSize((w, h))
                    _, raw_faces = smart_detector.detector.detect(img)
                    
                    # Restore original threshold
                    smart_detector.detector.setScoreThreshold(original_threshold)
                    
                    if raw_faces is not None and len(raw_faces) > 0:
                        logger.info(f"✅ Found {len(raw_faces)} faces with {attempt['description']}")
                        faces = raw_faces
                        break
                    else:
                        logger.warning(f"⚠️ No faces found with {attempt['description']}")
                        
                except Exception as e:
                    logger.error(f"❌ Detection attempt failed: {e}")
                    continue
        
        # Final check with proper numpy array handling
        if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
            logger.error(f"❌ No faces detected in: {Path(file_path).name}")
            logger.error("🔧 Debugging suggestions:")
            logger.error("   1. Check if image contains clear, visible faces")
            logger.error("   2. Verify faces are not too small (< 30px)")
            logger.error("   3. Try with different lighting/contrast")
            logger.error("   4. For ride photos, faces might be small or at angles")
            return []
        
        logger.info(f"✅ Found {len(faces)} faces in: {Path(file_path).name}")
        
        # Extract face embeddings
        results = []
        for i, face in enumerate(faces):
            try:
                x, y, w_box, h_box, confidence = face[:5]
                
                logger.debug(f"Processing face {i}: x={x:.1f}, y={y:.1f}, w={w_box:.1f}, h={h_box:.1f}, conf={confidence:.3f}")
                
                # Validate coordinates
                x1, y1 = max(int(x), 0), max(int(y), 0)
                x2, y2 = min(int(x + w_box), w), min(int(y + h_box), h)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"⚠️ Invalid face bounds for face {i}: ({x1},{y1}) to ({x2},{y2})")
                    continue
                
                # Check face size - more lenient for ride photos
                face_width = x2 - x1
                face_height = y2 - y1
                min_size = 20 if not is_selfie_mode else 30  # Smaller minimum for ride photos
                
                if face_width < min_size or face_height < min_size:
                    logger.warning(f"⚠️ Face {i} too small: {face_width}x{face_height}")
                    continue
                
                # Extract face region (convert BGR to RGB for consistency)
                face_crop_bgr = img[y1:y2, x1:x2]
                face_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                
                if face_crop.size == 0:
                    logger.warning(f"⚠️ Empty face crop for face {i}")
                    continue
                
                logger.debug(f"Face {i} crop size: {face_crop.shape}")
                
                # Generate embedding
                embedding = generate_face_embedding(face_crop)
                if embedding is None:
                    logger.warning(f"⚠️ Failed to generate embedding for face {i}")
                    continue
                
                # Create result
                result = {
                    'embedding': embedding.tolist(),
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(w_box), 'h': int(h_box)},
                    'confidence': float(confidence)
                }
                results.append(result)
                
                
                
            except Exception as e:
                logger.error(f"❌ Error processing face {i}: {e}")
                continue
        
        logger.info(f"🎯 Final result: {len(results)} faces processed from: {Path(file_path).name}")
        return results
        
    except Exception as e:
        logger.error(f"❌ SmartFaceDetector processing error for {file_path}: {e}")
        logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return []

def _fallback_face_detection(file_path: str) -> List[Dict[str, Any]]:
    """Fallback face detection using basic OpenCV methods"""
    try:
        logger.warning("🔄 Using fallback face detection")
        
        # Try basic OpenCV cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        logger.info(f"🔄 Fallback detected {len(faces)} faces")
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            try:
                # Extract face for embedding
                face_crop = img[y:y+h, x:x+w]
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Generate embedding
                embedding = generate_face_embedding(face_crop_rgb)
                if embedding is None:
                    continue
                
                result = {
                    'embedding': embedding.tolist(),
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'confidence': 0.8  # Default confidence for cascade
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Fallback processing error for face {i}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Fallback detection failed: {e}")
        return []

def generate_face_embedding(face_crop):
    """Generate face embedding using FaceNet"""
    try:
        # Resize to 160x160 for FaceNet
        face_img = cv2.resize(face_crop, (160, 160))
        
        # Convert to PIL Image and then to tensor
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

# ===== EXISTING CODE (with separated upload system) =====

class OptimizedFaceUploader:
    """High-performance face recognition uploader with separated data and file uploads"""
    
    def __init__(self, config: UploadConfig = None):
        self.config = config or UploadConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = threading.Lock()
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'OptimizedFaceUploader/2.0',
                'Accept': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def upload_batch_optimized(self, 
                                   upload_items: List[PhotoUploadItem],
                                   progress_callback=None) -> List[UploadResult]:
        """
        Main optimized upload method with separated data and file uploads
        
        Process:
        1. Upload all data (embeddings + metadata) first
        2. Upload files in parallel once data is ready
        3. Poll for completion status
        """
        total_items = len(upload_items)
        logger.info(f"🚀 Starting optimized batch upload: {total_items} items")
        
        if progress_callback:
            progress_callback("Starting optimized batch upload...", 0, total_items)
        
        start_time = time.time()
        
        try:
            # STEP 1: Upload all data first (fast)
            logger.info("📊 Step 1: Uploading face data and metadata...")
            if progress_callback:
                progress_callback("Uploading face data...", 0, total_items)
            
            data_upload_start = time.time()
            photo_ids = await self._upload_data_batch(upload_items, progress_callback)
            data_upload_time = time.time() - data_upload_start
            
            if not photo_ids:
                logger.error("❌ Data upload failed completely")
                return [UploadResult(success=False, error_message="Data upload failed") 
                       for _ in upload_items]
            
            logger.info(f"✅ Data upload completed in {data_upload_time:.2f}s: {len(photo_ids)} photos")
            
            # STEP 2: Upload files in parallel (slower)
            logger.info("📁 Step 2: Uploading files to storage...")
            if progress_callback:
                progress_callback("Uploading files...", len(photo_ids), total_items)
            
            file_upload_start = time.time()
            file_results = await self._upload_files_batch(upload_items, photo_ids, progress_callback)
            file_upload_time = time.time() - file_upload_start
            
            logger.info(f"✅ File upload completed in {file_upload_time:.2f}s")
            
            # STEP 3: Wait for processing completion and get final URLs
            logger.info("⏳ Step 3: Waiting for processing completion...")
            if progress_callback:
                progress_callback("Processing files...", len(photo_ids), total_items)
            
            status_check_start = time.time()
            final_results = await self._wait_for_completion(photo_ids, upload_items, progress_callback)
            status_check_time = time.time() - status_check_start
            
            total_time = time.time() - start_time
            successful = len([r for r in final_results if r.success])
            
            logger.info(f"🎯 Upload summary:")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Data upload: {data_upload_time:.2f}s")
            logger.info(f"   File upload: {file_upload_time:.2f}s") 
            logger.info(f"   Status checks: {status_check_time:.2f}s")
            logger.info(f"   Success rate: {successful}/{total_items} ({successful/total_items*100:.1f}%)")
            
            if progress_callback:
                progress_callback("Upload completed!", successful, total_items)
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Batch upload error: {e}")
            error_results = [
                UploadResult(
                    success=False, 
                    error_message=f"Batch upload failed: {str(e)}",
                    processing_time=time.time() - start_time
                ) for _ in upload_items
            ]
            return error_results
    
    async def _upload_data_batch(self, 
                               upload_items: List[PhotoUploadItem],
                               progress_callback=None) -> List[UUID]:
        """Upload face data and metadata in optimized batches"""
        
        # Split into batches for data upload
        batches = [
            upload_items[i:i + self.config.batch_size_data]
            for i in range(0, len(upload_items), self.config.batch_size_data)
        ]
        
        logger.info(f"📊 Uploading data in {len(batches)} batches")
        
        all_photo_ids = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_data)
        
        async def upload_data_batch(batch_items: List[PhotoUploadItem], batch_num: int):
            async with semaphore:
                try:
                    logger.info(f"📤 Uploading data batch {batch_num}: {len(batch_items)} items")
                    
                    # Prepare batch data
                    photos_data = []
                    for item in batch_items:
                        photo_data = {
                            "unit_code": item.unit_id,      # Send as unit_code
                            "outlet_code": item.outlet_id,  # Send as outlet_code
                            "photo_type_code": item.photo_type_id,  # Send as photo_type_code
                            "filename": item.filename,
                            "faces": [
                                {
                                    "embedding": face["embedding"],
                                    "bbox": face["bbox"],
                                    "confidence": face["confidence"]
                                }
                                for face in item.faces_data
                            ]
                        }
                        photos_data.append(photo_data)
                    
                    batch_payload = {"photos": photos_data}
                    
                    # Upload data
                    url = f"{self.config.api_base_url}/faces/upload-data"
                    
                    async with self.session.post(
                        url,
                        json=batch_payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_data)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            photo_ids = result.get("photo_ids", [])
                            
                            logger.info(f"✅ Data batch {batch_num} uploaded: {len(photo_ids)} photos")
                            
                            if progress_callback:
                                progress_callback(
                                    f"Data batch {batch_num} completed",
                                    len(all_photo_ids) + len(photo_ids),
                                    len(upload_items)
                                )
                            
                            return photo_ids
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Data batch {batch_num} failed: {response.status} - {error_text}")
                            return []
                            
                except Exception as e:
                    logger.error(f"❌ Data batch {batch_num} error: {e}")
                    return []
        
        # Execute all batch uploads concurrently
        tasks = [
            upload_data_batch(batch_items, i + 1)
            for i, batch_items in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all photo IDs
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                all_photo_ids.extend(batch_result)
        
        logger.info(f"📊 Data upload completed: {len(all_photo_ids)} total photo IDs")
        return all_photo_ids
    
    async def _upload_files_batch(self, 
                                upload_items: List[PhotoUploadItem],
                                photo_ids: List[UUID],
                                progress_callback=None) -> List[Dict]:
        """Upload files to storage in optimized batches"""
        
        if len(upload_items) != len(photo_ids):
            logger.error(f"❌ Mismatch: {len(upload_items)} items vs {len(photo_ids)} photo IDs")
            return []
        
        # Split into batches for file upload
        batch_size = self.config.batch_size_files
        batches = []
        
        for i in range(0, len(upload_items), batch_size):
            batch_items = upload_items[i:i + batch_size]
            batch_photo_ids = photo_ids[i:i + batch_size]
            batches.append((batch_items, batch_photo_ids))
        
        logger.info(f"📁 Uploading files in {len(batches)} batches")
        
        all_results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        async def upload_file_batch(batch_items: List[PhotoUploadItem], 
                                  batch_photo_ids: List[UUID], 
                                  batch_num: int):
            async with semaphore:
                try:
                    logger.info(f"📁 Uploading file batch {batch_num}: {len(batch_items)} files")
                    
                    # Prepare multipart data
                    data = aiohttp.FormData()
                    
                    # Add photo IDs
                    for photo_id in batch_photo_ids:
                        data.add_field('photo_ids', str(photo_id))
                    
                    # Add files
                    for item in batch_items:
                        try:
                            async with aiofiles.open(item.file_path, 'rb') as f:
                                file_content = await f.read()
                                data.add_field(
                                    'files',
                                    file_content,
                                    filename=item.filename,
                                    content_type='image/jpeg'
                                )
                        except Exception as e:
                            logger.error(f"❌ Failed to read file {item.file_path}: {e}")
                            continue
                    
                    # Upload files
                    url = f"{self.config.api_base_url}/faces/upload-files"
                    
                    async with self.session.post(
                        url,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_files)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ File batch {batch_num} uploaded successfully")
                            
                            if progress_callback:
                                progress_callback(
                                    f"File batch {batch_num} completed",
                                    len(all_results) + len(result),
                                    len(upload_items)
                                )
                            
                            return result
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ File batch {batch_num} failed: {response.status} - {error_text}")
                            return []
                            
                except Exception as e:
                    logger.error(f"❌ File batch {batch_num} error: {e}")
                    return []
        
        # Execute all file batch uploads concurrently
        tasks = [
            upload_file_batch(batch_items, batch_photo_ids, i + 1)
            for i, (batch_items, batch_photo_ids) in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all results
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                all_results.extend(batch_result)
        
        logger.info(f"📁 File upload completed: {len(all_results)} responses")
        return all_results
    
    async def _wait_for_completion(self, 
                                 photo_ids: List[UUID],
                                 upload_items: List[PhotoUploadItem],
                                 progress_callback=None) -> List[UploadResult]:
        """Wait for all uploads to complete and get final URLs"""
        
        logger.info(f"⏳ Monitoring completion status for {len(photo_ids)} photos")
        
        final_results = []
        pending_ids = set(str(pid) for pid in photo_ids)
        check_count = 0
        
        while pending_ids and check_count < self.config.max_status_checks:
            check_count += 1
            logger.info(f"🔍 Status check {check_count}: {len(pending_ids)} pending")
            
            try:
                # Check status in batches
                url = f"{self.config.api_base_url}/faces/batch-upload-status"
                check_ids = list(pending_ids)
                
                async with self.session.post(
                    url,
                    json={"photo_ids": check_ids},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        photos_status = result.get("photos", [])
                        
                        newly_completed = []
                        
                        for photo_status in photos_status:
                            photo_id = photo_status["photo_id"]
                            status = photo_status["status"]
                            
                            if status == "completed" :
                                # Photo is done
                                upload_result = UploadResult(
                                    success=True,
                                    photo_id=UUID(photo_id),
                                    original_url=photo_status.get("original_url"),
                                    thumbnail_url=photo_status.get("thumbnail_url"),
                                    processing_time=0.0  # Could track this better
                                )
                                final_results.append(upload_result)
                                newly_completed.append(photo_id)
                                
                            elif status == "failed":
                                # Photo failed
                                upload_result = UploadResult(
                                    success=False,
                                    photo_id=UUID(photo_id),
                                    error_message="File processing failed"
                                )
                                final_results.append(upload_result)
                                newly_completed.append(photo_id)
                        
                        # Remove completed photos from pending
                        for photo_id in newly_completed:
                            pending_ids.discard(photo_id)
                        
                        if newly_completed:
                            logger.info(f"✅ {len(newly_completed)} photos completed")
                            
                            if progress_callback:
                                progress_callback(
                                    f"Processing: {len(final_results)} completed",
                                    len(final_results),
                                    len(photo_ids)
                                )
                    
                    else:
                        logger.warning(f"⚠️ Status check failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Status check error: {e}")
            
            # Wait before next check
            if pending_ids:
                await asyncio.sleep(self.config.status_check_interval)
        
        # Handle any remaining pending photos as failures
        for photo_id in pending_ids:
            upload_result = UploadResult(
                success=False,
                photo_id=UUID(photo_id),
                error_message="Processing timeout"
            )
            final_results.append(upload_result)
        
        logger.info(f"⏳ Status monitoring completed: {len(final_results)} total results")
        return final_results

# ===== HIGH-LEVEL MANAGER =====

class FaceRecognitionUploadManager:
    """High-level manager for face recognition uploads"""
    
    def __init__(self, api_base_url: str = API_BASE):
        self.config = UploadConfig(api_base_url=api_base_url)
    
    async def upload_photos_optimized(self, 
                                    photos_data: List[Dict[str, Any]],
                                    progress_callback=None) -> List[UploadResult]:
        """
        Upload photos with optimized separated data/file approach
        
        Args:
            photos_data: List of photo data dictionaries containing:
                - file_path: str
                - unit_id: str
                - outlet_id: str
                - photo_type_id: str
                - faces_data: List[Dict] (embeddings, bboxes, confidences)
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of UploadResult objects
        """
        
        # Convert to PhotoUploadItem objects
        upload_items = []
        for photo_data in photos_data:
            try:
                item = PhotoUploadItem(
                    file_path=photo_data['file_path'],
                    unit_id=photo_data['unit_id'],
                    outlet_id=photo_data['outlet_id'],
                    photo_type_id=photo_data['photo_type_id'],
                    faces_data=photo_data['faces_data']
                )
                upload_items.append(item)
            except KeyError as e:
                logger.error(f"❌ Invalid photo data: missing {e}")
                continue
        
        if not upload_items:
            logger.error("❌ No valid upload items")
            return []
        
        # Use optimized uploader
        async with OptimizedFaceUploader(self.config) as uploader:
            results = await uploader.upload_batch_optimized(upload_items, progress_callback)
        
        return results
    
    def upload_photos_sync(self, 
                          photos_data: List[Dict[str, Any]],
                          progress_callback=None) -> List[UploadResult]:
        """Synchronous wrapper for async upload"""
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.upload_photos_optimized(photos_data, progress_callback)
        )

# ===== COMPATIBILITY FUNCTIONS =====

def batch_upload_to_backend_separated(files_data_list: List[Dict[str, Any]], 
                                     api_base_url: str = API_BASE,
                                     progress_callback=None) -> Tuple[bool, str]:
    """
    Drop-in replacement for existing batch_upload_to_backend_optimized function
    
    Args:
        files_data_list: List of file data dictionaries
        api_base_url: API base URL
        progress_callback: Optional progress callback
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    
    try:
        logger.info(f"🚀 Starting separated batch upload: {len(files_data_list)} files")
        
        # Convert existing format to new format
        photos_data = []
        for file_data in files_data_list:
            photo_data = {
                'file_path': file_data['file_path'],
                'unit_id': file_data.get('unit_id') or file_data.get('unit_code'),
                'outlet_id': file_data.get('outlet_id') or file_data.get('outlet_code'),
                'photo_type_id': file_data.get('photo_type_id') or file_data.get('photo_type_code'),
                'faces_data': file_data['faces']
            }
            photos_data.append(photo_data)
        
        # Use upload manager
        manager = FaceRecognitionUploadManager(api_base_url)
        results = manager.upload_photos_sync(photos_data, progress_callback)
        
        # Convert results to legacy format
        successful = len([r for r in results if r.success])
        total = len(results)
        
        if successful > 0:
            success_rate = (successful / total) * 100
            message = f"Separated upload: {successful}/{total} successful ({success_rate:.1f}%)"
            logger.info(f"✅ {message}")
            return True, message
        else:
            message = f"Separated upload failed: 0/{total} successful"
            logger.error(f"❌ {message}")
            return False, message
            
    except Exception as e:
        logger.error(f"❌ Separated batch upload error: {e}")
        return False, f"Upload failed: {str(e)}"

# ===== WORKER CLASS FOR COMPATIBILITY =====

from PyQt5.QtCore import QRunnable, pyqtSignal, QObject

class OptimizedBatchFaceEmbeddingWorkerSignals(QObject):
    """Signals for the worker"""
    finished = pyqtSignal(str, bool, str)  # result_summary, success, message
    progress = pyqtSignal(str, str)  # current_file, status
    error = pyqtSignal(str, str)  # file_path, error_message
    batch_completed = pyqtSignal(int, int)  # successful_count, failed_count

class OptimizedBatchFaceEmbeddingWorker(QRunnable):
    """
    Worker class that maintains compatibility with existing Explorer Window
    but uses the new separated upload system internally
    """
    
    def __init__(self, files_list: List[str], allowed_paths: List[str], max_upload_batch_size: int = 15):
        super().__init__()
        self.files_list = files_list
        self.allowed_paths = allowed_paths
        self.max_upload_batch_size = max_upload_batch_size
        self.signals = OptimizedBatchFaceEmbeddingWorkerSignals()
        
    def run(self):
        """Run the batch processing using separated upload system"""
        try:
            thread_name = f"Worker-{len(self.files_list)}"
            logger.info(f"🚀 [{thread_name}] Starting batch processing: {len(self.files_list)} files")
            
            self.signals.progress.emit("batch", f"🔄 Processing {len(self.files_list)} files...")
            
            # Progress callback for internal operations
            def progress_callback(message: str, current: int, total: int):
                self.signals.progress.emit("batch", f"🚀 {message}")
            
            # Process all files and extract face data
            photos_data = []
            processing_errors = 0
            
            for i, file_path in enumerate(self.files_list):
                try:
                    filename = Path(file_path).name
                    self.signals.progress.emit(file_path, f"🔍 Processing faces...")
                    
                    # Process faces in image (use ride mode, not selfie mode)
                    faces = process_faces_in_image_optimized(file_path, is_selfie_mode=False)
                    
                    if not faces:
                        processing_errors += 1
                        self.signals.error.emit(file_path, "No faces detected")
                        continue
                    
                    # Parse path codes
                    relative_path = self._get_relative_path(file_path)
                    if not relative_path:
                        processing_errors += 1
                        self.signals.error.emit(file_path, "Invalid path")
                        continue
                    
                    unit_code, outlet_code, photo_type_code = self._parse_codes_from_path(relative_path)
                    print(f"Parsed codes: unit={unit_code}, outlet={outlet_code}, type={photo_type_code}")
                    if not all([unit_code, outlet_code, photo_type_code]):
                        processing_errors += 1
                        self.signals.error.emit(file_path, "Path parsing failed")
                        continue
                    
                    # Convert codes to IDs (implement proper resolution based on your system)
                    unit_id, outlet_id, photo_type_id = self._resolve_codes_to_ids(
                        unit_code, outlet_code, photo_type_code
                    )
                    
                    photo_data = {
                        'file_path': file_path,
                        'unit_id': unit_id,
                        'outlet_id': outlet_id,
                        'photo_type_id': photo_type_id,
                        'faces_data': faces
                    }
                    photos_data.append(photo_data)
                    
                    self.signals.progress.emit(file_path, f"✅ {len(faces)} faces detected")
                    
                except Exception as e:
                    processing_errors += 1
                    self.signals.error.emit(file_path, f"Processing error: {str(e)}")
            
            if not photos_data:
                self.signals.finished.emit("No valid photos to upload", False, "All files failed processing")
                return
            
            # Use separated upload system
            self.signals.progress.emit("batch", "🚀 Starting separated upload...")
            
            manager = FaceRecognitionUploadManager(API_BASE)
            results = manager.upload_photos_sync(photos_data, progress_callback)
            
            # Process results
            successful = len([r for r in results if r.success])
            failed = len([r for r in results if not r.success])
            
            success_rate = (successful / len(results)) * 100 if results else 0
            
            if successful > 0:
                message = f"Separated upload: {successful}/{len(results)} successful ({success_rate:.1f}%)"
                self.signals.finished.emit(message, True, message)
            else:
                message = f"Upload failed: 0/{len(results)} successful"
                self.signals.finished.emit(message, False, message)
            
            self.signals.batch_completed.emit(successful, failed)
            
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
            self.signals.error.emit("batch", f"Worker error: {str(e)}")
            self.signals.finished.emit("Batch error", False, str(e))
    
    def _get_relative_path(self, file_path: str) -> Optional[str]:
        """Get relative path from allowed paths"""
        file_path_obj = Path(file_path).resolve()
        
        for root in self.allowed_paths:
            root_path = Path(root).resolve()
            try:
                relative = file_path_obj.relative_to(root_path)
                return str(relative)
            except ValueError:
                continue
        
        return None
    
    def _parse_codes_from_path(self, relative_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse codes from relative path"""
        try:
            parts = Path(relative_path).parts
            if len(parts) < 4:
                return None, None, None

            unit_code = parts[0].split("_")[0]
            outlet_code = parts[2].split("_")[0]
            photo_type_code = parts[1].split("_")[0]

            return unit_code, outlet_code, photo_type_code
        except Exception as e:
            logger.error(f"❌ Path parsing error: {e}")
            return None, None, None
    
    def _resolve_codes_to_ids(self, unit_code: str, outlet_code: str, photo_type_code: str) -> Tuple[str, str, str]:
        """
        Convert codes to UUIDs
        
        TODO: Implement proper code-to-ID resolution based on your system.
        For now, returning codes as IDs (you'll need to implement proper resolution)
        """
        logger.debug(f"🔧 Code resolution: {unit_code} -> {outlet_code} -> {photo_type_code}")
        
        # TEMPORARY: Return codes as IDs
        # In production, you should resolve these to actual UUIDs
        return unit_code, outlet_code, photo_type_code

# ===== BACKWARD COMPATIBILITY =====

def batch_upload_to_backend_optimized(files_data_list: List[Dict[str, Any]], 
                                     db_session=None,
                                     max_retries: int = 3,
                                     max_batch_size: int = 50,
                                     **kwargs) -> Tuple[bool, str]:
    """
    Backward compatibility function - redirects to separated upload system
    
    This maintains the same function signature as your existing code
    but uses the new separated upload system internally.
    """
    logger.info("🔄 Using separated upload system (backward compatibility mode)")
    
    # Extract API base URL from kwargs or use default
    api_base_url = kwargs.get('api_base_url', API_BASE)
    progress_callback = kwargs.get('progress_callback')
    
    return batch_upload_to_backend_separated(
        files_data_list, 
        api_base_url, 
        progress_callback
    )

# ===== UTILITY FUNCTIONS =====

def parse_codes_from_relative_path(relative_path: str, allowed_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse codes from relative path (compatibility function)"""
    try:
        parts = Path(relative_path).parts
        if len(parts) < 4:
            logger.warning(f"❌ Incomplete path: {relative_path}")
            return None, None, None

        unit_code = parts[0].split("_")[0]
        outlet_code = parts[1].split("_")[0]
        photo_type_code = parts[2].split("_")[0]

        return unit_code, outlet_code, photo_type_code
        
    except Exception as e:
        logger.error(f"❌ Path parsing error: {e}")
        return None, None, None

def get_relative_path(file_path: str, allowed_paths: List[str]) -> Optional[str]:
    """Get relative path (compatibility function)"""
    try:
        file_path_obj = Path(file_path).resolve()
        
        for root in allowed_paths:
            root_path = Path(root).resolve()
            try:
                relative = file_path_obj.relative_to(root_path)
                return str(relative)
            except ValueError:
                continue
        
        logger.warning(f"❌ File not in allowed paths: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Relative path error: {e}")
        return None

# ===== COMPATIBILITY WRAPPER FOR EXISTING BATCH PROCESSING =====

def process_batch_faces_and_upload_optimized(files_list: List[str], 
                                           allowed_paths: List[str], 
                                           db_session=None,
                                           max_upload_batch_size: int = 50) -> Tuple[bool, str]:
    """
    Compatibility function that maintains the existing function signature
    but uses the new separated upload system internally.
    
    This allows your existing code to work without changes while
    getting the benefits of the separated upload system.
    """
    
    try:
        logger.info(f"🚀 Processing batch with separated upload: {len(files_list)} files")
        
        # Process all files and extract face data
        photos_data = []
        processing_errors = 0
        
        for file_path in files_list:
            try:
                # Process faces in image (use ride mode for attraction photos)
                faces = process_faces_in_image_optimized(file_path, is_selfie_mode=False)
                
                if not faces:
                    processing_errors += 1
                    logger.warning(f"⚠️ No faces detected in {file_path}")
                    continue
                
                # Parse path codes
                relative_path = get_relative_path(file_path, allowed_paths)
                if not relative_path:
                    processing_errors += 1
                    logger.warning(f"⚠️ Invalid path for {file_path}")
                    continue
                
                unit_code, outlet_code, photo_type_code = parse_codes_from_relative_path(
                    relative_path, allowed_paths[0]
                )
                
                if not all([unit_code, outlet_code, photo_type_code]):
                    processing_errors += 1
                    logger.warning(f"⚠️ Path parsing failed for {file_path}")
                    continue
                
                # Convert codes to IDs (you'll need to implement proper resolution)
                unit_id, outlet_id, photo_type_id = unit_code, outlet_code, photo_type_code
                
                photo_data = {
                    'file_path': file_path,
                    'unit_id': unit_id,
                    'outlet_id': outlet_id,
                    'photo_type_id': photo_type_id,
                    'faces_data': faces
                }
                photos_data.append(photo_data)
                
                logger.info(f"✅ Processed {Path(file_path).name}: {len(faces)} faces")
                
            except Exception as e:
                processing_errors += 1
                logger.error(f"❌ Processing error for {file_path}: {e}")
        
        if not photos_data:
            message = f"No valid photos to upload. {processing_errors} processing errors."
            logger.error(f"❌ {message}")
            return False, message
        
        logger.info(f"📊 Ready to upload {len(photos_data)} photos ({processing_errors} errors)")
        
        # Use separated upload system
        manager = FaceRecognitionUploadManager(API_BASE)
        results = manager.upload_photos_sync(photos_data)
        
        # Process results
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if not r.success])
        
        success_rate = (successful / len(results)) * 100 if results else 0
        
        if successful > 0:
            message = f"Batch upload: {successful}/{len(results)} successful ({success_rate:.1f}%), {processing_errors} processing errors"
            logger.info(f"✅ {message}")
            return True, message
        else:
            message = f"Batch upload failed: 0/{len(results)} successful, {processing_errors} processing errors"
            logger.error(f"❌ {message}")
            return False, message
            
    except Exception as e:
        logger.error(f"❌ Batch processing error: {e}")
        return False, f"Batch processing failed: {str(e)}"

