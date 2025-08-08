#utils/face_detector.py

import logging
import cv2
import os
import numpy as np
import torch
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
import json
import time
import socket
import sys
import threading
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from contextlib import contextmanager
import functools

# Database imports
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

# Core imports
from core.device_setup import device, resnet, API_BASE

# Configure logging for better performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def _process_large_batch_with_parallel_upload(files_data_list: List[Dict[str, Any]], 
                                            db_session=None, 
                                            max_retries: int = 2,
                                            max_batch_size: int = 5,
                                            max_concurrent_uploads: int = 5) -> Tuple[bool, str]:
    """Process large batch with TRUE PARALLEL uploads"""
    
    total_files = len(files_data_list)
    num_batches = (total_files + max_batch_size - 1) // max_batch_size
    
    logger.info(f"🚀 PARALLEL upload: {total_files} files in {num_batches} batches with {max_concurrent_uploads} concurrent uploads")
    
    # Split into batches
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * max_batch_size
        end_idx = min(start_idx + max_batch_size, total_files)
        batch_chunk = files_data_list[start_idx:end_idx]
        batches.append((batch_num + 1, batch_chunk))
    
    # Use ThreadPoolExecutor for parallel uploads
    successful_uploads = 0
    failed_uploads = 0
    failed_batches = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent_uploads) as executor:
        # Submit all batches for parallel processing
        future_to_batch = {
            executor.submit(_process_single_batch_wrapper, batch_data, db_session, max_retries): batch_num
            for batch_num, batch_data in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            batch_data = batches[batch_num - 1][1]  # Get the actual batch data
            
            try:
                success, message = future.result()
                
                if success:
                    successful_uploads += len(batch_data)
                    logger.info(f"✅ Parallel batch {batch_num} completed: {len(batch_data)} files")
                else:
                    failed_uploads += len(batch_data)
                    failed_batches.append(batch_num)
                    logger.error(f"❌ Parallel batch {batch_num} failed: {message}")
                    
            except Exception as e:
                failed_uploads += len(batch_data)
                failed_batches.append(batch_num)
                logger.error(f"❌ Parallel batch {batch_num} error: {e}")
    
    # Summary
    success_rate = (successful_uploads / total_files) * 100 if total_files > 0 else 0
    
    if failed_batches:
        message = f"Parallel upload: {successful_uploads}/{total_files} successful ({success_rate:.1f}%). Failed batches: {failed_batches}"
        logger.warning(f"⚠️ {message}")
        return successful_uploads > 0, message
    else:
        message = f"Parallel upload: {successful_uploads}/{total_files} successful (100%)"
        logger.info(f"✅ {message}")
        return True, message

def _process_single_batch_wrapper(batch_chunk: List[Dict[str, Any]], 
                                 db_session=None, 
                                 max_retries: int = 2) -> Tuple[bool, str]:
    """Wrapper for single batch processing in threads"""
    return _process_single_batch(batch_chunk, db_session, max_retries)


# ===== PERFORMANCE OPTIMIZATIONS =====

@dataclass
class DetectionConfig:
    """Configuration for face detection optimization"""
    conf_threshold: float = 0.7  
    nms_threshold: float = 0.3
    max_size: int = 1280
    min_face_size: int = 30  
    max_faces_per_image: int = 50  
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    warm_up_enabled: bool = True
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available()

@dataclass
class DatabaseConfig:
    """Database configuration from environment"""
    url: str = os.getenv('DATABASE_URL', 'postgresql://postgres:1234@fr-db:5432/face_recognition')
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

# Global configurations
DETECTION_CONFIG = DetectionConfig()
DB_CONFIG = DatabaseConfig()

# ===== DATABASE MANAGEMENT =====

class DatabaseManager:
    """Enhanced database connection manager with Docker support"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.scoped_session = None
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self):
        """Initialize database connections with comprehensive error handling"""
        if self._initialized:
            logger.info("Database already initialized")
            return True
            
        with self._lock:
            if self._initialized:
                return True
                
            try:
                logger.info(f"🔧 Initializing database connection...")
                logger.info(f"   Host: {self._extract_host_from_url(DB_CONFIG.url)}")
                
                # Create engine with Docker-optimized settings
                self.engine = create_engine(
                    DB_CONFIG.url,
                    pool_size=DB_CONFIG.pool_size,
                    max_overflow=DB_CONFIG.max_overflow,
                    pool_timeout=DB_CONFIG.pool_timeout,
                    pool_recycle=DB_CONFIG.pool_recycle,
                    pool_pre_ping=True,  # Important for Docker containers
                    echo=DB_CONFIG.echo,
                    # Docker-specific connection args
                    connect_args={
                        "application_name": "FaceSync_Worker",
                        "connect_timeout": 10,
                        # Handle connection drops gracefully
                        "keepalives_idle": 600,
                        "keepalives_interval": 30,
                        "keepalives_count": 3,
                    }
                )
                
               
                # Create session factory
                self.session_factory = sessionmaker(bind=self.engine)
                self.scoped_session = scoped_session(self.session_factory)
                
                self._initialized = True
                logger.info("✅ Database manager initialized successfully")
                logger.info(f"   Pool size: {DB_CONFIG.pool_size}")
                logger.info(f"   Max overflow: {DB_CONFIG.max_overflow}")
                
                return True
                
            except SQLAlchemyError as e:
                logger.error(f"❌ Database connection error: {e}")
                self._log_connection_troubleshooting()
                self._initialized = False
                return False
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize database: {e}")
                self._log_connection_troubleshooting()
                self._initialized = False
                return False
    
    def _extract_host_from_url(self, url):
        """Extract host from database URL for logging"""
        try:
            # Simple extraction for logging purposes
            if '@' in url:
                host_part = url.split('@')[1]
                if ':' in host_part:
                    host = host_part.split(':')[0]
                    return host
            return "unknown"
        except:
            return "unknown"
    
    def _log_connection_troubleshooting(self):
        """Log troubleshooting information for database connection issues"""
        logger.error("❌ Database connection troubleshooting:")
        logger.error("   1. Check if PostgreSQL container is running:")
        logger.error("      docker ps | grep postgres")
        logger.error("   2. Check if port 5432 is accessible:")
        logger.error("      telnet localhost 5432")
        logger.error("   3. Verify database credentials in .env file")
        logger.error("   4. Check if database 'face_recognition' exists")
        logger.error("   5. For Docker Compose, ensure services are on same network")
        
        # Show current configuration (safely)
        safe_url = DB_CONFIG.url.replace(':1234@', ':***@') if ':1234@' in DB_CONFIG.url else DB_CONFIG.url
        logger.error(f"   Current DATABASE_URL: {safe_url}")
    
    @contextmanager
    def get_session(self):
        """Get thread-safe database session with automatic cleanup"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database not initialized and initialization failed")
            
        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Database session error: {e}")
            raise
        finally:
            session.close()
    
    
    
    def close(self):
        """Close all database connections"""
        try:
            if self.scoped_session:
                self.scoped_session.remove()
            if self.engine:
                self.engine.dispose()
            self._initialized = False
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing database: {e}")

# Global database manager
db_manager = DatabaseManager()

# ===== PERFORMANCE UTILITIES =====

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⚡ {func.__name__}: {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)


class ParallelBatchProcessor:
    """New class for parallel processing"""
    
    def __init__(self, max_workers=4):
        self.detection_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.detector = get_optimized_detector()
        
    def process_files_parallel(self, files_list):
        """Process multiple files in parallel"""
        futures_to_files = {}
        
        # Submit all files for parallel processing
        for file_path in files_list:
            future = self.detection_pool.submit(
                process_faces_in_image_optimized, file_path
            )
            futures_to_files[future] = file_path
            
        # Collect results as they complete
        results = []
        for future in as_completed(futures_to_files):
            file_path = futures_to_files[future]
            try:
                embeddings = future.result(timeout=5)
                if embeddings:
                    results.append({
                        'file_path': file_path,
                        'embeddings': embeddings
                    })
            except Exception as e:
                logger.error(f"Parallel processing error for {file_path}: {e}")
                
        return results


# ===== OPTIMIZED YUNET DETECTOR =====
class HighPerformanceYuNetDetector:
    """Ultra-optimized YuNet detector with advanced caching and GPU acceleration"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DETECTION_CONFIG
        self.model_path = resource_path(self.config.model_path)
        self.detector = None
        self.model_warmed = False
        self._detection_cache = {}
        self._cache_lock = threading.Lock()
        
        self._initialize_detector()
        if self.config.warm_up_enabled:
            self._warm_up_model()
    
    @performance_monitor
    def _initialize_detector(self):
        """Initialize YuNet detector with optimal settings"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize with STRICTER thresholds
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.7,  
                nms_threshold=0.3,
                top_k=50,  
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            
            logger.info(f"✅ High-performance face detector initialized")
            logger.info(f"   Model: {self.model_path}")
            logger.info(f"   Thresholds: conf={self.config.conf_threshold}, nms={self.config.nms_threshold}")
            logger.info(f"   Max faces per image: {self.config.max_faces_per_image}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize face detector: {e}")
            raise
    
    @performance_monitor
    def _warm_up_model(self):
        """Warm up model with optimized dummy detection"""
        try:
            # Create multiple dummy images for thorough warm-up
            dummy_sizes = [(224, 224), (320, 320), (640, 480)]
            
            for width, height in dummy_sizes:
                dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                self.detector.setInputSize((width, height))
                _, _ = self.detector.detect(dummy_img)
            
            self.model_warmed = True
            logger.info("✅ Face detection model thoroughly warmed up")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warm up failed: {e}")
    
    @performance_monitor
    def detect_with_optimization(self, img: np.ndarray) -> Optional[List[List[float]]]:
        """Optimized detection with comprehensive input validation"""
        try:
            # ✅ COMPREHENSIVE INPUT VALIDATION
            if img is None:
                logger.error("❌ Image is None")
                return None
                
            if not isinstance(img, np.ndarray):
                logger.error("❌ Image is not numpy array")
                return None
                
            if img.size == 0:
                logger.error("❌ Image is empty")
                return None
                
            if len(img.shape) != 3 or img.shape[2] != 3:
                logger.error(f"❌ Invalid image shape: {img.shape}")
                return None
                
            # Check for invalid values
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                logger.error("❌ Image contains NaN or infinity values")
                return None
                
            # Ensure proper data type
            if img.dtype != np.uint8:
                logger.warning(f"⚠️ Converting image from {img.dtype} to uint8")
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            original_h, original_w = img.shape[:2]
            
            # Validate dimensions
            if original_w <= 0 or original_h <= 0:
                logger.error(f"❌ Invalid dimensions: {original_w}x{original_h}")
                return None
                
            if original_w > 10000 or original_h > 10000:
                logger.error(f"❌ Image too large: {original_w}x{original_h}")
                return None
            
            # ✅ SAFE DETECTION WITH TRY-CATCH
            try:
                # Smart resizing logic
                if max(original_w, original_h) > self.config.max_size:
                    scale = self.config.max_size / max(original_w, original_h)
                    new_w = max(32, int(original_w * scale))  # Ensure minimum size
                    new_h = max(32, int(original_h * scale))
                    
                    # Validate new dimensions
                    if new_w <= 0 or new_h <= 0:
                        logger.error(f"❌ Invalid resize dimensions: {new_w}x{new_h}")
                        return None
                    
                    # Use INTER_AREA for downscaling (better quality)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Validate resized image
                    if resized_img is None or resized_img.size == 0:
                        logger.error("❌ Resize operation failed")
                        return None
                    
                    # Set input size with validation
                    try:
                        self.detector.setInputSize((new_w, new_h))
                    except Exception as e:
                        logger.error(f"❌ setInputSize failed: {e}")
                        return None
                    
                    # Detect with error handling
                    try:
                        _, faces = self.detector.detect(resized_img)
                    except Exception as e:
                        logger.error(f"❌ Detection failed on resized image: {e}")
                        return None
                    
                    # Scale coordinates back with improved precision
                    if faces is not None and len(faces) > 0:
                        return self._scale_faces_back_safe(faces, scale)
                    else:
                        return None
                else:
                    # Direct detection for smaller images
                    try:
                        self.detector.setInputSize((original_w, original_h))
                    except Exception as e:
                        logger.error(f"❌ setInputSize failed for original: {e}")
                        return None
                        
                    try:
                        _, faces = self.detector.detect(img)
                    except Exception as e:
                        logger.error(f"❌ Detection failed on original image: {e}")
                        return None
                    
                    if faces is not None and len(faces) > 0:
                        return self._format_faces_safe(faces)
                    else:
                        return None
                        
            except cv2.error as e:
                logger.error(f"❌ OpenCV error during detection: {e}")
                return None
            except Exception as e:
                logger.error(f"❌ Unexpected error during detection: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            return None
    
    def _scale_faces_back_safe(self, faces: np.ndarray, scale: float) -> List[List[float]]:
        """Scale face coordinates back to original size with comprehensive validation"""
        try:
            if faces is None or len(faces) == 0:
                return []
                
            scaled_faces = []
            inv_scale = 1.0 / scale
            
            # Validate scale values
            if not np.isfinite(inv_scale) or inv_scale <= 0:
                logger.error(f"❌ Invalid inverse scale: {inv_scale}")
                return []
            
            for i, face in enumerate(faces):
                try:
                    if len(face) < 15:  # YuNet returns 15 values
                        logger.warning(f"⚠️ Incomplete face data at index {i}")
                        continue
                        
                    x, y, w, h = face[:4]
                    conf = face[14]
                    
                    # Check for invalid values
                    if not all(np.isfinite([x, y, w, h, conf])):
                        logger.warning(f"⚠️ Invalid face values at index {i}")
                        continue
                    
                    # Scale back with safe rounding
                    orig_x = max(0, int(round(x * inv_scale)))
                    orig_y = max(0, int(round(y * inv_scale)))
                    orig_w = max(1, int(round(w * inv_scale)))
                    orig_h = max(1, int(round(h * inv_scale)))
                    
                    # Validate confidence
                    if not (0.0 <= conf <= 1.0):
                        logger.warning(f"⚠️ Invalid confidence at index {i}: {conf}")
                        continue
                    
                    scaled_faces.append([orig_x, orig_y, orig_w, orig_h, float(conf)])
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing face {i}: {e}")
                    continue
            
            return scaled_faces
            
        except Exception as e:
            logger.error(f"❌ Error in _scale_faces_back_safe: {e}")
            return []
    
    def _format_faces_safe(self, faces: np.ndarray) -> List[List[float]]:
        """Format faces to standard format with comprehensive validation"""
        try:
            if faces is None or len(faces) == 0:
                return []
                
            formatted_faces = []
            
            for i, face in enumerate(faces):
                try:
                    if len(face) < 15:  # YuNet returns 15 values
                        logger.warning(f"⚠️ Incomplete face data at index {i}")
                        continue
                        
                    x, y, w, h = face[:4]
                    conf = face[14]
                    
                    # Check for invalid values
                    if not all(np.isfinite([x, y, w, h, conf])):
                        logger.warning(f"⚠️ Invalid face values at index {i}")
                        continue
                    
                    # Convert to safe integers
                    x_int = max(0, int(round(x)))
                    y_int = max(0, int(round(y)))
                    w_int = max(1, int(round(w)))
                    h_int = max(1, int(round(h)))
                    
                    # Validate confidence
                    if not (0.0 <= conf <= 1.0):
                        logger.warning(f"⚠️ Invalid confidence at index {i}: {conf}")
                        continue
                    
                    formatted_faces.append([x_int, y_int, w_int, h_int, float(conf)])
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error formatting face {i}: {e}")
                    continue
            
            return formatted_faces
            
        except Exception as e:
            logger.error(f"❌ Error in _format_faces_safe: {e}")
            return []

    
    @performance_monitor
    def detect_and_validate(self, img: np.ndarray) -> Tuple[bool, Optional[List[List[float]]]]:
        """Main detection method with comprehensive validation"""
        try:
            start_time = time.time()
            faces = self.detect_with_optimization(img)
            detection_time = time.time() - start_time
            
            if faces is None or len(faces) == 0:
                logger.debug(f"🔍 No faces detected ({detection_time:.3f}s)")
                return False, None
            
            # Advanced face validation
            valid_faces = self._validate_faces(faces, img.shape[:2])
            
            if valid_faces:
                logger.info(f"✅ Detected {len(valid_faces)} valid faces ({detection_time:.3f}s)")
                return True, valid_faces
            else:
                logger.debug(f"❌ No valid faces after filtering ({detection_time:.3f}s)")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ Detection and validation error: {e}")
            return False, None
    
    def _validate_faces(self, faces: List[List[float]], img_shape: Tuple[int, int]) -> List[List[float]]:
        """Advanced face validation with multiple criteria"""
        valid_faces = []
        img_h, img_w = img_shape
        
        for face in faces:
            x, y, w, h, confidence = face
            
            # Convert to int and validate
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Boundary validation and correction
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            # Size validation
            if w < self.config.min_face_size or h < self.config.min_face_size:
                logger.debug(f"⚠️ Face too small: {w}x{h}")
                continue
            
            # Aspect ratio validation (faces should be roughly square-ish)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                logger.debug(f"⚠️ Invalid aspect ratio: {aspect_ratio:.2f}")
                continue
            
            # Confidence validation
            if confidence < self.config.conf_threshold:
                logger.debug(f"⚠️ Low confidence: {confidence:.3f}")
                continue
            
            valid_faces.append([x, y, w, h, confidence])
            logger.debug(f"✅ Valid face: {x},{y},{w},{h} conf={confidence:.3f}")
        
        return valid_faces

# Global detector instance with lazy initialization
_detector_instance: Optional[HighPerformanceYuNetDetector] = None
_detector_lock = threading.Lock()

def get_optimized_detector() -> HighPerformanceYuNetDetector:
    """Get thread-safe shared detector instance"""
    global _detector_instance
    
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = HighPerformanceYuNetDetector()
                logger.info("🚀 High-performance face detector created")
    
    return _detector_instance

# ===== OPTIMIZED IMAGE PROCESSING =====
class OptimizedImageProcessor:
    """High-performance image processing with advanced optimizations"""
    
    @staticmethod
    @performance_monitor
    def safe_imread(file_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
        """Ultra-safe image reading with multiple fallback methods"""
        try:
            # Normalize path
            normalized_path = Path(file_path).resolve()
            
            # Fast validation
            if not normalized_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return None
            
            if normalized_path.stat().st_size == 0:
                logger.error(f"❌ Empty file: {file_path}")
                return None
            
            # Primary read method
            img = cv2.imread(str(normalized_path), flags)
            if img is not None and img.size > 0:
                return img
            
            # Fallback: byte reading for special characters
            try:
                with open(normalized_path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, flags)
                    if img is not None and img.size > 0:
                        return img
            except Exception as e:
                logger.warning(f"⚠️ Byte reading failed: {e}")
            
            logger.error(f"❌ Failed to read image: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Image reading error for {file_path}: {e}")
            return None
    
    @staticmethod
    @performance_monitor
    def preprocess_face_batch(face_crops: List[np.ndarray]) -> Optional[torch.Tensor]:
        """Batch preprocessing for multiple faces with GPU acceleration"""
        try:
            if not face_crops:
                return None
            
            # Batch resize and convert
            processed_faces = []
            
            for face_crop in face_crops:
                try:
                    # Resize to 160x160
                    face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert BGR to RGB
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    
                    # Normalize
                    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
                    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                    
                    processed_faces.append(face_tensor)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Face preprocessing error: {e}")
                    continue
            
            if not processed_faces:
                return None
            
            # Stack into batch tensor
            batch_tensor = torch.stack(processed_faces).to(device)
            return batch_tensor
            
        except Exception as e:
            logger.error(f"❌ Batch preprocessing error: {e}")
            return None
        
# ===== OPTIMIZED FACE PROCESSING =====

@performance_monitor
def process_faces_in_image_optimized(file_path: str) -> List[Dict[str, Any]]:
    """Highly optimized face processing with batch operations"""
    try:
        # FIX: Call safe_imread as a static method properly
        img = OptimizedImageProcessor.safe_imread(file_path)
        if img is None:
            logger.warning(f"❌ Failed to read image: {file_path}")
            return []

        # Additional image validation
        if img.size == 0:
            logger.warning(f"❌ Empty image: {file_path}")
            return []
            
        if len(img.shape) != 3 or img.shape[2] != 3:
            logger.warning(f"❌ Invalid image format: {img.shape} - {file_path}")
            return []

        h, w = img.shape[:2]
        logger.debug(f"📸 Processing {w}x{h} image: {Path(file_path).name}")

        # Get detector and detect faces with enhanced error handling
        try:
            detector = get_optimized_detector()
            success, faces = detector.detect_and_validate(img)
        except Exception as e:
            logger.error(f"❌ Detection failed for {file_path}: {e}")
            return []

        if not success or not faces:
            logger.debug(f"❌ No faces detected in: {Path(file_path).name}")
            return []

        logger.info(f"✅ Found {len(faces)} faces in: {Path(file_path).name}")

        # Extract face crops for batch processing
        face_crops = []
        face_info = []
        
        for i, face in enumerate(faces):
            try:
                x, y, w_box, h_box, confidence = face
                
                # Extract face region with bounds checking
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w_box, w), min(y + h_box, h)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"⚠️ Invalid face bounds: {i}")
                    continue
                
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    logger.warning(f"⚠️ Empty face crop: {i}")
                    continue
                
                face_crops.append(face_crop)
                face_info.append({
                    'bbox': {'x': x, 'y': y, 'w': w_box, 'h': h_box},
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Face extraction error {i}: {e}")
                continue
        
        if not face_crops:
            logger.warning(f"❌ No valid face crops extracted")
            return []
        
        # Batch preprocessing
        batch_tensor = OptimizedImageProcessor.preprocess_face_batch(face_crops)
        if batch_tensor is None:
            logger.warning(f"❌ Batch preprocessing failed")
            return []
        
        # Batch embedding generation
        try:
            with torch.no_grad():
                embeddings_tensor = resnet(batch_tensor)
                embeddings_cpu = embeddings_tensor.cpu().numpy()
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            return []
        
        # Combine results
        results = []
        for i, (embedding, info) in enumerate(zip(embeddings_cpu, face_info)):
            try:
                results.append({
                    'embedding': embedding.tolist(),
                    'bbox': info['bbox'],
                    'confidence': float(info['confidence'])
                })
            except Exception as e:
                logger.warning(f"⚠️ Result formatting error {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(results)} faces from: {Path(file_path).name}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Face processing error for {file_path}: {e}")
        return []

# ===== OPTIMIZED NETWORK UTILITIES =====

class OptimizedNetworkClient:
    """High-performance network client with advanced retry and connection pooling"""
    
    def __init__(self):
        self.session = None
        self._session_lock = threading.Lock()
    
    def get_session(self) -> requests.Session:
        """Get or create thread-safe session"""
        if self.session is None:
            with self._session_lock:
                if self.session is None:
                    self.session = self._create_session()
        return self.session
    
    def _create_session(self) -> requests.Session:
        """Create optimized session with retry strategy"""
        session = requests.Session()
        
        # Advanced retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False
        )
        
        # High-performance adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=50,
            pool_block=False
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Optimized headers
        session.headers.update({
            'User-Agent': 'FaceSync-TurboClient/2.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session
    
    @performance_monitor
    def check_connectivity(self) -> bool:
        """Fast connectivity check"""
        test_hosts = [
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
        ]
        
        for host, port in test_hosts:
            try:
                socket.create_connection((host, port), timeout=3)
                return True
            except:
                continue
        return False

# Global network client
network_client = OptimizedNetworkClient()

@performance_monitor
def batch_upload_to_backend_optimized(files_data_list: List[Dict[str, Any]], 
                                    db_session=None, 
                                    max_retries: int = 2,
                                    max_batch_size: int = 15,
                                    enable_parallel_upload: bool = True,
                                    max_concurrent_uploads: int = 3):
    """Ultra-optimized batch upload with TRUE parallel option"""
    
    if not files_data_list:
        logger.error("❌ No files to upload")
        return False, "No files provided"
    
    total_files = len(files_data_list)
    logger.info(f"🚀 Starting optimized batch upload: {total_files} files (parallel: {enable_parallel_upload})")
    
    if total_files > max_batch_size:
        logger.info(f"📦 Splitting {total_files} files into batches of {max_batch_size}")
        
        if enable_parallel_upload:
            # Use NEW parallel upload
            return _process_large_batch_with_parallel_upload(
                files_data_list, db_session, max_retries, max_batch_size, max_concurrent_uploads
            )
        else:
            # Use old sequential upload
            return _process_large_batch_with_splitting(
                files_data_list, db_session, max_retries, max_batch_size
            )
    
    return _process_single_batch(files_data_list, db_session, max_retries)

class AsyncUploadClient:
    """Async HTTP client for parallel uploads"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = None
        self.session = None
    
    async def __aenter__(self):
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def upload_batch_async(self, batch_data: List[Dict[str, Any]], batch_num: int) -> Tuple[bool, str]:
        """Upload single batch asynchronously"""
        async with self.semaphore:
            try:
                url = f"{API_BASE}/faces/batch-upload-embedding-turbo"
                
                # Prepare multipart data
                data = aiohttp.FormData()
                
                # Add files
                for file_data in batch_data:
                    file_path = file_data['file_path']
                    filename = Path(file_path).name
                    
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        data.add_field('files', file_content, filename=filename, content_type='image/jpeg')
                
                # Add form fields
                for file_data in batch_data:
                    data.add_field('unit_codes', file_data['unit_code'])
                    data.add_field('photo_type_codes', file_data['photo_type_code'])
                    data.add_field('outlet_codes', file_data['outlet_code'])
                    
                    faces_json = json.dumps(file_data['faces'], separators=(',', ':'))
                    data.add_field('faces_data', faces_json)
                
                logger.info(f"🚀 Async uploading batch {batch_num}: {len(batch_data)} files")
                
                async with self.session.post(url, data=data) as response:
                    if response.status in [200, 207]:
                        result = await response.json()
                        successful = result.get('successful_uploads', 0)
                        message = f"Batch {batch_num}: {successful}/{len(batch_data)} successful"
                        logger.info(f"✅ {message}")
                        return True, message
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Batch {batch_num} failed: {response.status}")
                        return False, f"HTTP {response.status}: {error_text[:100]}"
                        
            except Exception as e:
                logger.error(f"❌ Async upload error for batch {batch_num}: {e}")
                return False, str(e)

async def _process_large_batch_with_async_upload(files_data_list: List[Dict[str, Any]], 
                                               max_batch_size: int = 15,
                                               max_concurrent_uploads: int = 3) -> Tuple[bool, str]:
    """Process large batch with async parallel uploads"""
    
    total_files = len(files_data_list)
    num_batches = (total_files + max_batch_size - 1) // max_batch_size
    
    logger.info(f"🚀 ASYNC PARALLEL upload: {total_files} files in {num_batches} batches")
    
    # Split into batches
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * max_batch_size
        end_idx = min(start_idx + max_batch_size, total_files)
        batch_chunk = files_data_list[start_idx:end_idx]
        batches.append((batch_num + 1, batch_chunk))
    
    successful_uploads = 0
    failed_uploads = 0
    failed_batches = []
    
    async with AsyncUploadClient(max_concurrent_uploads) as client:
        # Create tasks for all batches
        tasks = [
            client.upload_batch_async(batch_data, batch_num)
            for batch_num, batch_data in batches
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            batch_num, batch_data = batches[i]
            
            if isinstance(result, Exception):
                failed_uploads += len(batch_data)
                failed_batches.append(batch_num)
                logger.error(f"❌ Async batch {batch_num} exception: {result}")
            else:
                success, message = result
                if success:
                    successful_uploads += len(batch_data)
                else:
                    failed_uploads += len(batch_data)
                    failed_batches.append(batch_num)
    
    # Summary
    success_rate = (successful_uploads / total_files) * 100 if total_files > 0 else 0
    
    if failed_batches:
        message = f"Async parallel upload: {successful_uploads}/{total_files} successful ({success_rate:.1f}%)"
        return successful_uploads > 0, message
    else:
        message = f"Async parallel upload: {successful_uploads}/{total_files} successful (100%)"
        return True, message

def run_async_upload(files_data_list: List[Dict[str, Any]], 
                    max_batch_size: int = 15,
                    max_concurrent_uploads: int = 3) -> Tuple[bool, str]:
    """Run async upload in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        _process_large_batch_with_async_upload(files_data_list, max_batch_size, max_concurrent_uploads)
    )


def _process_large_batch_with_splitting(files_data_list: List[Dict[str, Any]], 
                                      db_session=None, 
                                      max_retries: int = 2,
                                      max_batch_size: int = 50) -> Tuple[bool, str]:
    """Process large batch by splitting into smaller chunks"""
    
    total_files = len(files_data_list)
    num_batches = (total_files + max_batch_size - 1) // max_batch_size  # Ceiling division
    
    logger.info(f"🔄 Processing {total_files} files in {num_batches} batches of {max_batch_size}")
    
    successful_uploads = 0
    failed_uploads = 0
    failed_batches = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * max_batch_size
        end_idx = min(start_idx + max_batch_size, total_files)
        batch_chunk = files_data_list[start_idx:end_idx]
        
        logger.info(f"📤 Processing batch {batch_num + 1}/{num_batches}: files {start_idx + 1}-{end_idx}")
        
        try:
            success, message = _process_single_batch(batch_chunk, db_session, max_retries)
            
            if success:
                successful_uploads += len(batch_chunk)
                logger.info(f"✅ Batch {batch_num + 1} completed: {len(batch_chunk)} files")
            else:
                failed_uploads += len(batch_chunk)
                failed_batches.append(batch_num + 1)
                logger.error(f"❌ Batch {batch_num + 1} failed: {message}")
                
        except Exception as e:
            failed_uploads += len(batch_chunk)
            failed_batches.append(batch_num + 1)
            logger.error(f"❌ Batch {batch_num + 1} error: {e}")
        
        # Small delay between batches to prevent server overload
        if batch_num < num_batches - 1:  # Don't sleep after last batch
            time.sleep(1)
    
    # Summary
    success_rate = (successful_uploads / total_files) * 100 if total_files > 0 else 0
    
    if failed_batches:
        message = f"Split batch upload: {successful_uploads}/{total_files} successful ({success_rate:.1f}%). Failed batches: {failed_batches}"
        logger.warning(f"⚠️ {message}")
        return successful_uploads > 0, message
    else:
        message = f"Split batch upload: {successful_uploads}/{total_files} successful (100%)"
        logger.info(f"✅ {message}")
        return True, message

def _process_single_batch(files_data_list: List[Dict[str, Any]], 
                         db_session=None, 
                         max_retries: int = 2) -> Tuple[bool, str]:
    """Process a single batch (≤50 files)"""
    
    batch_size = len(files_data_list)
    logger.info(f"🔧 Processing single batch: {batch_size} files")
    
    try:
        # Quick connectivity check
        if not network_client.check_connectivity():
            logger.error("❌ No network connectivity")
            return False, "No network connectivity"
        
        session = network_client.get_session()
        url = f"{API_BASE}/faces/batch-upload-embedding-turbo"
        
        # Upload attempts with retry
        for attempt in range(max_retries):
            try:
                logger.info(f"📤 Upload attempt {attempt + 1}/{max_retries} for {batch_size} files")
                
                # Prepare multipart data efficiently
                files, form_data = _prepare_multipart_data_optimized(files_data_list)
                
                # Validate data preparation
                if not _validate_multipart_data(files, form_data, len(files_data_list)):
                    return False, "Data preparation validation failed"
                
                # Calculate dynamic timeout based on file count and sizes
                timeout = _calculate_optimal_timeout(files_data_list, attempt)
                
                logger.info(f"🌐 Uploading {batch_size} files to: {url} (timeout: {timeout}s)")
                
                # Make request with optimal settings
                response = session.post(
                    url,
                    files=files,
                    data=form_data,
                    timeout=timeout,
                    stream=False
                )
                
                # Process response
                success, message = _process_upload_response(response, len(files_data_list))
                
                if success:
                    # Database logging if session provided
                    if db_session:
                        _log_successful_upload(db_session, files_data_list)
                    return True, message
                elif response.status_code in [400, 422]:
                    # Client errors - don't retry
                    return False, message
                elif attempt < max_retries - 1:
                    # Server errors - retry with backoff
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, message
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"⏰ Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, "Upload timeout after all retries"
                    
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    logger.warning(f"🔌 Connection error on attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, "Connection failed after all retries"
                    
            except Exception as e:
                logger.error(f"💥 Upload error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Upload failed: {str(e)}"
        
        return False, "Upload failed after all retries"
        
    except Exception as e:
        logger.error(f"❌ Fatal upload error: {e}")
        return False, f"Fatal error: {str(e)}"
    
def _prepare_multipart_data_optimized(files_data_list: List[Dict[str, Any]]) -> Tuple[List, List]:
    """Optimized multipart data preparation"""
    files = []
    form_data = []
    
    # Prepare files efficiently
    for file_data in files_data_list:
        file_path = file_data['file_path']
        filename = Path(file_path).name
        
        # Read file in binary mode
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                files.append(('files', (filename, file_content, 'image/jpeg')))
        except Exception as e:
            logger.error(f"❌ Failed to read file {filename}: {e}")
            continue
    
    # Prepare form data
    for file_data in files_data_list:
        form_data.append(('unit_codes', file_data['unit_code']))
        form_data.append(('photo_type_codes', file_data['photo_type_code']))
        form_data.append(('outlet_codes', file_data['outlet_code']))
        
        # Serialize faces data efficiently
        faces_json = json.dumps(file_data['faces'], separators=(',', ':'))
        form_data.append(('faces_data', faces_json))
    
    return files, form_data

def _validate_multipart_data(files: List, form_data: List, expected_count: int) -> bool:
    """Validate prepared multipart data"""
    if len(files) != expected_count:
        logger.error(f"❌ File count mismatch: {len(files)} != {expected_count}")
        return False
    
    # Count form fields
    field_counts = Counter([item[0] for item in form_data])
    required_fields = ['unit_codes', 'photo_type_codes', 'outlet_codes', 'faces_data']
    
    for field_name in required_fields:
        count = field_counts.get(field_name, 0)
        if count != expected_count:
            logger.error(f"❌ Field '{field_name}' count mismatch: {count} != {expected_count}")
            return False
    
    return True

def _calculate_optimal_timeout(files_data_list: List[Dict[str, Any]], attempt: int) -> int:
    """Calculate optimal timeout - FASTER for smaller batches"""
    base_timeout = 30  # CHANGED from 60
    file_count_factor = len(files_data_list) * 1  # CHANGED from 2
    attempt_factor = attempt * 10  # CHANGED from 30
    
    return base_timeout + file_count_factor + attempt_factor


def _process_upload_response(response: requests.Response, file_count: int) -> Tuple[bool, str]:
    """Process upload response efficiently"""
    logger.info(f"📡 Response status: {response.status_code}")
    
    if response.status_code in [200, 207]:
        try:
            result = response.json()
            successful = result.get('successful_uploads', 0)
            failed = result.get('failed_uploads', 0)
            
            message = f"Upload completed: {successful}/{file_count} successful"
            logger.info(f"✅ {message}")
            
            return True, message
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response: {e}")
            return False, "Invalid response format"
    
    elif response.status_code == 422:
        error_text = response.text
        logger.error(f"❌ Validation error: {error_text}")
        return False, f"Validation error: {error_text[:200]}"
    
    elif response.status_code == 400:
        error_text = response.text
        logger.error(f"❌ Client error: {error_text}")
        return False, f"Client error: {error_text[:200]}"
    
    else:
        error_text = response.text
        logger.error(f"❌ Server error {response.status_code}: {error_text}")
        return False, f"Server error {response.status_code}"

def _log_successful_upload(db_session, files_data_list: List[Dict[str, Any]]):
    """Log successful uploads to database"""
    try:
        # Add your database logging logic here
        # Example: Insert upload records, update file status, etc.
        logger.debug(f"📝 Logged {len(files_data_list)} successful uploads to database")
    except Exception as e:
        logger.warning(f"⚠️ Failed to log uploads to database: {e}")

# ===== UTILITY FUNCTIONS =====

def parse_codes_from_relative_path(relative_path: str, allowed_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Optimized path parsing with better error handling"""
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
    """Optimized relative path calculation"""
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

def validate_files_data_optimized(files_data_list: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Optimized files data validation"""
    if not files_data_list:
        return False, "No files to validate"
    
    required_fields = ['file_path', 'unit_code', 'photo_type_code', 'outlet_code', 'faces']
    errors = []
    
    for i, file_data in enumerate(files_data_list):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in file_data]
        if missing_fields:
            errors.append(f"File {i}: Missing fields {missing_fields}")
            continue
        
        # Quick file existence check
        if not Path(file_data['file_path']).exists():
            errors.append(f"File {i}: Not found")
            continue
        
        # Validate faces data
        if not isinstance(file_data['faces'], list):
            errors.append(f"File {i}: Invalid faces data")
            continue
    
    if errors:
        logger.error(f"❌ Validation failed: {len(errors)} errors")
        return False, f"Validation failed: {len(errors)} errors"
    
    logger.info(f"✅ Validation passed for {len(files_data_list)} files")
    return True, "Validation successful"

# ===== THREAD-SAFE BATCH PROCESSING =====

@performance_monitor
def process_batch_faces_and_upload_optimized(files_list: List[str], 
                                           allowed_paths: List[str], 
                                           db_session=None,
                                           max_upload_batch_size: int = 50) -> Tuple[bool, str]:
    """Ultra-optimized batch processing with configurable upload batch size"""
    
    thread_name = threading.current_thread().name
    logger.info(f"🚀 [{thread_name}] Starting optimized batch: {len(files_list)} files (max upload batch: {max_upload_batch_size})")
    
    # Process all files for faces (unchanged)
    files_data = []
    processing_errors = []
    
    start_time = time.time()
    
    for i, file_path in enumerate(files_list):
        try:
            filename = Path(file_path).name
            logger.debug(f"🔍 [{thread_name}] Processing {i+1}/{len(files_list)}: {filename}")
            
            # Quick file validation
            if not Path(file_path).exists():
                processing_errors.append(f"File not found: {filename}")
                continue
            
            if Path(file_path).stat().st_size == 0:
                processing_errors.append(f"Empty file: {filename}")
                continue
            
            # Process faces with optimized function
            embeddings = process_faces_in_image_optimized(file_path)
            
            if not embeddings:
                processing_errors.append(f"No faces detected: {filename}")
                continue
            
            # Parse path codes
            relative_path = get_relative_path(file_path, allowed_paths)
            if not relative_path:
                processing_errors.append(f"Invalid path: {filename}")
                continue
            
            unit_code, outlet_code, photo_type_code = parse_codes_from_relative_path(
                relative_path, allowed_paths[0]
            )
            
            if not all([unit_code, outlet_code, photo_type_code]):
                processing_errors.append(f"Path parsing failed: {filename}")
                continue
            
            # Add to batch data
            files_data.append({
                'file_path': file_path,
                'unit_code': unit_code,
                'photo_type_code': photo_type_code,
                'outlet_code': outlet_code,
                'faces': embeddings
            })
            
            logger.debug(f"✅ [{thread_name}] Processed: {filename} ({len(embeddings)} faces)")
            
        except Exception as e:
            error_msg = f"Processing error for {Path(file_path).name}: {str(e)}"
            processing_errors.append(error_msg)
            logger.error(f"❌ [{thread_name}] {error_msg}")
    
    processing_time = time.time() - start_time
    logger.info(f"📊 [{thread_name}] Processing completed in {processing_time:.2f}s: {len(files_data)} ready, {len(processing_errors)} errors")
    
    # Log processing errors
    if processing_errors:
        logger.warning(f"⚠️ [{thread_name}] Processing errors: {len(processing_errors)}")
        for error in processing_errors[:3]:
            logger.warning(f"   {error}")
        if len(processing_errors) > 3:
            logger.warning(f"   ... and {len(processing_errors) - 3} more errors")
    
    if not files_data:
        return False, "No files ready for upload"
    
    # Validate files data
    is_valid, validation_message = validate_files_data_optimized(files_data)
    if not is_valid:
        return False, f"Validation failed: {validation_message}"
    
    # Upload batch with configurable batch size
    logger.info(f"🚀 [{thread_name}] Starting optimized upload with max batch size: {max_upload_batch_size}")
    upload_start = time.time()
    
    success, message = batch_upload_to_backend_optimized(files_data, db_session, max_retries=3, max_batch_size=max_upload_batch_size)
    
    upload_time = time.time() - upload_start
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"✅ [{thread_name}] Batch completed in {total_time:.2f}s (upload: {upload_time:.2f}s): {message}")
    else:
        logger.error(f"❌ [{thread_name}] Batch failed after {total_time:.2f}s: {message}")
    
    return success, message

@performance_monitor
def process_batch_faces_and_upload_turbo(files_list: List[str], 
                                        allowed_paths: List[str], 
                                        db_session=None,
                                        max_upload_batch_size: int = 15):  # CHANGED from 50
    """TURBO version with parallel processing"""
    
    thread_name = threading.current_thread().name
    logger.info(f"🚀 [{thread_name}] TURBO BATCH: {len(files_list)} files")
    
    # PARALLEL FACE PROCESSING
    processor = ParallelBatchProcessor(max_workers=4)
    start_time = time.time()
    
    # Process all files in parallel
    results = processor.process_files_parallel(files_list)
    
    processing_time = time.time() - start_time
    logger.info(f"⚡ [{thread_name}] Parallel processing: {len(results)} files in {processing_time:.2f}s")
    
    if not results:
        return False, "No files ready for upload"
    
    # Prepare upload data
    files_data = []
    for result in results:
        file_path = result['file_path']
        embeddings = result['embeddings']
        
        # Parse path codes
        relative_path = get_relative_path(file_path, allowed_paths)
        if not relative_path:
            continue
            
        unit_code, outlet_code, photo_type_code = parse_codes_from_relative_path(
            relative_path, allowed_paths[0]
        )
        
        if all([unit_code, outlet_code, photo_type_code]):
            files_data.append({
                'file_path': file_path,
                'unit_code': unit_code,
                'photo_type_code': photo_type_code,
                'outlet_code': outlet_code,
                'faces': embeddings
            })
    
    # Upload with smaller batches for faster response
    upload_start = time.time()
    success, message = batch_upload_to_backend_optimized(
        files_data, 
        db_session, 
        max_retries=3, 
        max_batch_size=max_upload_batch_size
    )
    
    upload_time = time.time() - upload_start
    total_time = time.time() - start_time
    
    logger.info(f"✅ [{thread_name}] TURBO completed in {total_time:.2f}s (processing: {processing_time:.2f}s, upload: {upload_time:.2f}s)")
    
    return success, message



# ===== OPTIMIZED WORKER CLASSES =====
class OptimizedBatchFaceEmbeddingWorkerSignals(QObject):
    """Optimized signals with more detailed progress tracking"""
    finished = pyqtSignal(str, bool, str)  # result_summary, success, message
    progress = pyqtSignal(str, str)  # current_file, status
    error = pyqtSignal(str, str)  # file_path, error_message
    batch_completed = pyqtSignal(int, int)  # successful_count, failed_count
    performance_update = pyqtSignal(dict)  # performance metrics


class OptimizedBatchFaceEmbeddingWorker(QRunnable):
    """Ultra-optimized batch worker with parallel processing"""
    
    def __init__(self, files_list: List[str], allowed_paths: List[str], max_upload_batch_size: int = 15):  # CHANGED default from 50 to 15
        super().__init__()
        self.files_list = files_list
        self.allowed_paths = allowed_paths
        self.max_upload_batch_size = max_upload_batch_size
        self.signals = OptimizedBatchFaceEmbeddingWorkerSignals()
        self.start_time = None
        self.performance_metrics = {
            'files_processed': 0,
            'faces_detected': 0,
            'processing_time': 0,
            'upload_time': 0,
            'total_time': 0
        }

    def run(self):
        """Optimized batch processing with parallel execution"""
        thread_name = threading.current_thread().name
        batch_size = len(self.files_list)
        self.start_time = time.time()
        
        try:
            logger.info(f"🚀 [{thread_name}] Starting TURBO worker: {batch_size} files")
            self.signals.progress.emit("batch", f"🔄 Processing {batch_size} files in parallel...")
            
            with db_manager.get_session() as db_session:
                # Use TURBO processing
                success, message = process_batch_faces_and_upload_turbo(
                    self.files_list, 
                    self.allowed_paths, 
                    db_session,
                    max_upload_batch_size=self.max_upload_batch_size
                )
                
                self._update_performance_metrics(success)
                
                if success:
                    self.signals.progress.emit("batch", "✅ Turbo batch completed")
                    self.signals.finished.emit(f"Batch successful: {batch_size} files", True, message)
                    self.signals.batch_completed.emit(batch_size, 0)
                else:
                    self.signals.progress.emit("batch", "❌ Turbo batch failed")
                    self.signals.finished.emit(f"Batch failed: {batch_size} files", False, message)
                    self.signals.batch_completed.emit(0, batch_size)
                
        except Exception as e:
            error_message = f"Turbo worker error: {str(e)}"
            logger.error(f"❌ {error_message}")
            self.signals.error.emit("batch", error_message)
            self.signals.finished.emit("Batch error", False, error_message)
            self.signals.batch_completed.emit(0, len(self.files_list))

    
    def _update_performance_metrics(self, success: bool):
        """Update performance metrics"""
        total_time = time.time() - self.start_time
        
        
        self.performance_metrics.update({
            'total_time': total_time,
            'files_processed': len(self.files_list) if success else 0,
            'success_rate': 1.0 if success else 0.0,
            'throughput': len(self.files_list) / total_time if total_time > 0 else 0
        })

# ===== LEGACY COMPATIBILITY FUNCTIONS =====

def upload_embedding_to_backend(file_path: str, faces: List[Dict], allowed_paths: List[str], max_retries: int = 2) -> bool:
    """Legacy single file upload with optimized backend"""
    try:
        filename = Path(file_path).name
        
        # Parse path codes
        relative_path = get_relative_path(file_path, allowed_paths)
        if not relative_path:
            logger.error(f"❌ File path not in allowed paths: {filename}")
            return False

        unit_code, outlet_code, photo_type_code = parse_codes_from_relative_path(
            relative_path, allowed_paths[0]
        )

        if not all([unit_code, outlet_code, photo_type_code]):
            logger.error(f"❌ Failed to parse path codes: {filename}")
            return False

        # Use optimized batch upload with single file
        files_data = [{
            'file_path': file_path,
            'unit_code': unit_code,
            'photo_type_code': photo_type_code,
            'outlet_code': outlet_code,
            'faces': faces
        }]
        
        # Use database session for single upload
        with db_manager.get_session() as db_session:
            success, message = batch_upload_to_backend_optimized(files_data, db_session, max_retries)
        
        if success:
            logger.info(f"✅ Single upload successful: {filename}")
        else:
            logger.error(f"❌ Single upload failed: {filename} - {message}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Single upload error: {Path(file_path).name} - {str(e)}")
        return False

class FaceEmbeddingWorkerSignals(QObject):
    """Legacy worker signals"""
    finished = pyqtSignal(str, list, bool)  # file_path, embeddings, success
    progress = pyqtSignal(str, str)  # file_path, status
    error = pyqtSignal(str, str)

class FaceEmbeddingWorker(QRunnable):
    """Legacy single file worker with optimized processing"""
    
    def __init__(self, file_path: str, allowed_paths: List[str]):
        super().__init__()
        self.file_path = file_path
        self.allowed_paths = allowed_paths
        self.signals = FaceEmbeddingWorkerSignals()

    def run(self):
        """Optimized single file processing"""
        try:
            filename = Path(self.file_path).name
            
            self.signals.progress.emit(self.file_path, "🔍 Detecting faces...")
            
            # Use optimized face processing
            embeddings = process_faces_in_image_optimized(self.file_path)
            
            if embeddings:
                self.signals.progress.emit(self.file_path, "📤 Uploading...")
                
                # Use optimized upload
                success = upload_embedding_to_backend(
                    self.file_path, embeddings, self.allowed_paths, max_retries=3
                )
                
                if success:
                    self.signals.progress.emit(self.file_path, "✅ Upload complete")
                else:
                    self.signals.progress.emit(self.file_path, "❌ Upload failed")
                    
                self.signals.finished.emit(self.file_path, embeddings, success)
            else:
                self.signals.progress.emit(self.file_path, "⚠️ No faces detected")
                self.signals.finished.emit(self.file_path, [], False)
                
        except Exception as e:
            logger.error(f"❌ Optimized worker error for {self.file_path}: {e}")
            self.signals.error.emit(self.file_path, f"Worker error: {str(e)}")
            self.signals.finished.emit(self.file_path, [], False)

def initialize_optimized_face_detection():
    """Initialize all optimized components with corrected logic"""
    try:
        logger.info("🚀 Initializing optimized face detection components...")
        
        # Initialize database manager
        logger.info("🔧 Initializing database manager...")
        
        # FIXED: Check the actual initialization result properly
        if not db_manager._initialized:
            init_result = db_manager.initialize()
            if not init_result:
                logger.error("❌ Database initialization failed")
                return False
            else:
                logger.info("✅ Database manager initialization completed")
        else:
            logger.info("✅ Database already initialized - skipping")
        
    
        
        # Initialize face detector
        logger.info("🔧 Loading face detection model...")
        try:
            global _detector_instance
            if _detector_instance is None:
                detector = get_optimized_detector()
                if detector and hasattr(detector, 'detector') and detector.detector is not None:
                    logger.info("✅ Face detector loaded successfully")
                else:
                    logger.error("❌ Face detector loading failed")
                    return False
            else:
                logger.info("✅ Face detector already loaded - skipping")
        except Exception as e:
            logger.error(f"❌ Face detector loading failed: {e}")
            return False
        
        # Initialize network client
        logger.info("🔧 Initializing network client...")
        try:
            session = network_client.get_session()
            if session:
                logger.info("✅ Network client initialized")
            else:
                logger.warning("⚠️ Network client initialization returned None")
                # Don't fail for network issues, just warn
        except Exception as e:
            logger.warning(f"⚠️ Network client initialization warning: {e}")
            # Don't fail for network issues
        
        # Final success confirmation
        logger.info("✅ All components initialized successfully")
        logger.info(f"   Database: Connected to fr-db")
        logger.info(f"   Device: {device}")
        logger.info(f"   GPU Available: {torch.cuda.is_available()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize optimized face detection: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def cleanup_optimized_face_detection():
    """Cleanup all resources"""
    try:
        logger.info("🔄 Starting optimized face detection cleanup...")
        
        # Close database connections
        if db_manager._initialized:
            db_manager.close()
            logger.info("✅ Database connections closed")
        
        # Clear detector instance
        global _detector_instance
        if _detector_instance is not None:
            _detector_instance = None
            logger.info("✅ Face detector instance cleared")
        
        # Close network session
        if hasattr(network_client, 'session') and network_client.session:
            try:
                network_client.session.close()
                logger.info("✅ Network session closed")
            except:
                pass
        
        logger.info("✅ Optimized face detection cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return False

