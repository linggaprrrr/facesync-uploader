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
from retinaface import RetinaFace
import time
import socket
from core.device_setup import device, resnet, API_BASE

# Shared detector instance untuk reuse
_detector_instance = None
logger = logging.getLogger(__name__)

class OptimizedRetinaFaceDetector:
    """Optimized RetinaFace detector dengan speed improvements"""
    
    def __init__(self, conf_threshold=0.6, nms_threshold=0.4, max_size=640, device=None):
        # Auto-detect device or use specified device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_size = max_size
        self.model_warmed = False
        
        # Print device info
        if self.device == 'cuda':
            print(f"✅ RetinaFace using CUDA - GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ RetinaFace using CPU")
        
        self._warm_up_model()
    
    def _warm_up_model(self):
        """Warm up model dengan dummy detection"""
        try:
            dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            RetinaFace.detect_faces(dummy_img, threshold=0.9)
            self.model_warmed = True
            logger.info("✅ RetinaFace model warmed up")
        except Exception as e:
            logger.warning(f"⚠️ Model warm up failed: {e}")
    
    def detect_with_resize(self, img):
        """Detect dengan image resizing dan FIXED coordinate scaling"""
        original_h, original_w = img.shape[:2]
        
        # Resize jika terlalu besar
        if max(original_w, original_h) > self.max_size:
            scale = self.max_size / max(original_w, original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            logger.info(f"🔄 Resizing: {original_w}x{original_h} -> {new_w}x{new_h} (scale={scale:.3f})")
            
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            faces_dict = RetinaFace.detect_faces(
                resized_img, 
                threshold=self.conf_threshold,
                model=None,
                allow_upscaling=False
            )
            
            # FIXED: Scale coordinates back properly
            if faces_dict:
                for face_key, face_data in faces_dict.items():
                    facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
                    
                    # Scale back ke original size
                    x1, y1, x2, y2 = facial_area
                    original_x1 = int(x1 / scale)
                    original_y1 = int(y1 / scale) 
                    original_x2 = int(x2 / scale)
                    original_y2 = int(y2 / scale)
                    
                    # Update dengan koordinat original
                    face_data['facial_area'] = [original_x1, original_y1, original_x2, original_y2]
                    
                    logger.debug(f"Scaled bbox: ({x1},{y1},{x2},{y2}) -> ({original_x1},{original_y1},{original_x2},{original_y2})")
        else:
            faces_dict = RetinaFace.detect_faces(
                img, 
                threshold=self.conf_threshold,
                model=None,
                allow_upscaling=False
            )
        
        return faces_dict
    
    def detect(self, img):
        """Main detection method dengan FIXED bbox conversion"""
        try:
            start_time = time.time()
            faces_dict = self.detect_with_resize(img)
            detection_time = time.time() - start_time
            
            logger.info(f"🔍 Detection time: {detection_time:.3f}s")
            
            if not faces_dict:
                return False, None
            
            faces_list = []
            img_h, img_w = img.shape[:2]
            
            for face_key, face_data in faces_dict.items():
                facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
                confidence = float(face_data['score'])
                
                # FIXED: Proper conversion dari [x1,y1,x2,y2] ke [x,y,w,h]
                x1, y1, x2, y2 = facial_area
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)  # ✅ width = x2 - x1
                h = int(y2 - y1)  # ✅ height = y2 - y1
                
                # Validasi bbox
                if w <= 0 or h <= 0:
                    logger.warning(f"⚠️ Invalid bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    continue
                
                # Pastikan bbox dalam bounds image
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))
                
                face_array = [x, y, w, h, confidence]
                faces_list.append(face_array)
                
                logger.debug(f"Face bbox: x={x}, y={y}, w={w}, h={h}, conf={confidence:.3f}")
            
            return True, faces_list
            
        except Exception as e:
            logger.error(f"❌ Error dalam deteksi: {e}")
            return False, None


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def get_shared_detector():
    """Get shared detector instance dengan GPU optimization"""
    global _detector_instance
    if _detector_instance is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _detector_instance = OptimizedRetinaFaceDetector(
            device=device,
            conf_threshold=0.6,
            nms_threshold=0.4,
            max_size=640  # Bisa naik ke 1024 jika pakai GPU untuk akurasi lebih tinggi
        )
        
        # Log GPU usage
        if device == 'cuda':
            logger.info(f"🚀 Face detector using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Face detector using CPU")
            
    return _detector_instance

def create_face_detector():
    """Factory function dengan shared instance"""
    return get_shared_detector()

def normalize_path(file_path):
    """Normalize file path to handle different path formats"""
    try:
        # Convert to Path object and resolve
        path = Path(file_path).resolve()
        return str(path)
    except Exception as e:
        logger.warning(f"Path normalization failed for {file_path}: {e}")
        return file_path


def validate_image_file(file_path):
    """Validate if the image file exists and is readable"""
    try:
        normalized_path = normalize_path(file_path)
        
        # Check if file exists
        if not os.path.exists(normalized_path):
            logger.error(f"❌ File tidak ditemukan: {normalized_path}")
            return False, normalized_path
            
        # Check if file is readable
        if not os.access(normalized_path, os.R_OK):
            logger.error(f"❌ File tidak dapat dibaca: {normalized_path}")
            return False, normalized_path
            
        # Check file size
        file_size = os.path.getsize(normalized_path)
        if file_size == 0:
            logger.error(f"❌ File kosong: {normalized_path}")
            return False, normalized_path
            
        return True, normalized_path
        
    except Exception as e:
        logger.error(f"❌ Error validating file {file_path}: {e}")
        return False, file_path
    
def safe_imread(file_path, flags=cv2.IMREAD_COLOR):
    """Safely read image with multiple fallback methods"""
    try:
        # First, validate the file
        is_valid, normalized_path = validate_image_file(file_path)
        if not is_valid:
            return None
            
        # Try reading with normalized path
        img = cv2.imread(normalized_path, flags)
        if img is not None:
            return img
            
        # Try with original path if normalization failed
        if normalized_path != file_path:
            img = cv2.imread(file_path, flags)
            if img is not None:
                return img
                
        # Try reading as bytes (for special characters in path)
        try:
            with open(normalized_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, flags)
                if img is not None:
                    return img
        except Exception as e:
            logger.warning(f"Byte reading failed: {e}")
            
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in safe_imread for {file_path}: {e}")
        return None
    
def process_faces_in_image(file_path, original_shape=None, pad=None, scale=None):
    """Optimized face processing dengan error handling yang lebih baik"""
    try:
        # Use safe image reading
        img = safe_imread(file_path)
        if img is None:
            logger.warning(f"❌ Gagal membaca gambar: {file_path}")
            return []

        h, w = img.shape[:2]
        logger.info(f"📸 Processing image: {w}x{h} - {file_path}")

        # Validate image dimensions
        if h == 0 or w == 0:
            logger.warning(f"❌ Invalid image dimensions: {w}x{h}")
            return []

        # Gunakan shared detector
        face_detector = get_shared_detector()
        success, faces = face_detector.detect(img)

        if not success or faces is None or len(faces) == 0:
            logger.warning("❌ Tidak ada wajah terdeteksi.")
            return []

        logger.info(f"✅ {len(faces)} wajah terdeteksi dengan RetinaFace.")

        embeddings = []
        for i, face in enumerate(faces):
            try:
                x, y, w_box, h_box = map(int, face[:4])
                confidence = float(face[4])
                
                # Validasi koordinat dengan bounds checking
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w_box, w), min(y + h_box, h)
                
                # Ensure minimum face size
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    logger.warning(f"⚠️ Face {i} too small or invalid bbox")
                    continue
                    
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    logger.warning(f"⚠️ Empty face crop for face {i}")
                    continue

                # Optimized preprocessing with error handling
                try:
                    face_crop_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                    face_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
                except Exception as resize_error:
                    logger.warning(f"⚠️ Resize error for face {i}: {resize_error}")
                    continue

                # Tensor conversion optimization dengan GPU support
                try:
                    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
                    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                    face_tensor = face_tensor.unsqueeze(0).to(device)  # Move to GPU

                    with torch.no_grad():
                        embedding_tensor = resnet(face_tensor).squeeze()
                        embedding = embedding_tensor.cpu().numpy().tolist()  # Move back to CPU for JSON
                        
                except Exception as tensor_error:
                    logger.warning(f"⚠️ Tensor processing error for face {i}: {tensor_error}")
                    continue

                # Bbox calculation
                if original_shape and pad and scale:
                    try:
                        bbox_dict = {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)}
                        original_bbox = reverse_letterbox(
                            bbox=bbox_dict,
                            original_shape=original_shape,
                            resized_shape=img.shape[:2],
                            pad=pad,
                            scale=scale
                        )
                        original_bbox = convert_to_json_serializable(original_bbox)
                    except Exception as bbox_error:
                        logger.warning(f"⚠️ Bbox conversion error: {bbox_error}")
                        original_bbox = {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)}
                else:
                    original_bbox = {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)}

                embeddings.append({
                    "embedding": embedding,
                    "bbox": original_bbox,
                    "confidence": confidence
                })

            except Exception as e:
                logger.warning(f"⚠️ Error processing face {i}: {e}")
                continue

        logger.info(f"✅ Successfully processed {len(embeddings)} faces from {file_path}")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Error processing image {file_path}: {e}")
        return []

# JSON utilities
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder untuk numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def safe_json_dumps(data):
    """Safe JSON serialization"""
    try:
        return json.dumps(data, cls=NumpyEncoder)
    except Exception as e:
        logger.error(f"❌ JSON serialization error: {e}")
        converted_data = convert_to_json_serializable(data)
        return json.dumps(converted_data)

def safe_json_loads(json_string):
    """Safe JSON deserialization"""
    try:
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"❌ JSON deserialization error: {e}")
        return None

def parse_codes_from_relative_path(relative_path, allowed_path):
    """Parse unit, outlet, photo_type codes from path"""
    try:
        parts = relative_path.split(os.sep)
        if len(parts) < 4:
            logger.warning(f"Path tidak lengkap: {relative_path}")
            return None, None, None

        unit_folder = parts[0]
        outlet_folder = parts[1]
        photo_type_folder = parts[2]

        unit_code = unit_folder.split("_")[0]
        outlet_code = outlet_folder.split("_")[0]
        photo_type_code = photo_type_folder.split("_")[0]

        return unit_code, outlet_code, photo_type_code
    except Exception as e:
        logger.error(f"Error parsing codes: {e}")
        return None, None, None

def get_relative_path(file_path, allowed_paths):
    """Get relative path from allowed paths"""
    try:
        file_path = os.path.abspath(file_path)
        for root in allowed_paths:
            root = os.path.abspath(root)
            if file_path.startswith(root):
                return os.path.relpath(file_path, root)
        return None
    except Exception as e:
        logger.error(f"Error getting relative path: {e}")
        return None

# ===== ENHANCED CONNECTION HANDLING =====

def create_robust_session():
    """Create session dengan retry strategy untuk upload"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,                    # Total retries
        backoff_factor=2,          # Wait time progression: 1s, 2s, 4s
        status_forcelist=[429, 500, 502, 503, 504],  # Server errors to retry
        allowed_methods=["POST"]    # Only retry POST for uploads
    )
    
    # Mount adapter dengan retry
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set headers
    session.headers.update({
        'User-Agent': 'FaceSync-Client/1.0',
        'Accept': 'application/json'
    })
    
    return session

def check_network_connectivity():
    """Quick network connectivity check"""
    test_hosts = [
        ("8.8.8.8", 53),       # Google DNS
        ("1.1.1.1", 53),       # Cloudflare DNS
    ]
    
    for host, port in test_hosts:
        try:
            socket.create_connection((host, port), timeout=5)
            return True
        except:
            continue
    return False

def check_server_health(api_base, timeout=10):
    """Check if server is responding"""
    try:
        # Try to reach server with a simple request
        test_url = f"{api_base}/health"  # Adjust sesuai endpoint server Anda
        # Jika tidak ada health endpoint, gunakan endpoint lain yang ringan
        
        response = requests.get(test_url, timeout=timeout)
        return response.status_code < 500
        
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception:
        # If no health endpoint, assume server is reachable if we get any response
        return True

def upload_embedding_to_backend(file_path, faces, allowed_paths, max_retries=3):
    """Upload embedding dengan enhanced error handling dan retry mechanism"""
    
    filename = os.path.basename(file_path)
    
    try:
        # Step 1: Basic validations
        relative_path = get_relative_path(file_path, allowed_paths)
        if not relative_path:
            logger.error(f"❌ File path tidak termasuk folder yang diizinkan: {filename}")
            return False

        unit_code, photo_type_code, outlet_code = parse_codes_from_relative_path(
            relative_path, allowed_paths[0]
        )

        if not all([unit_code, outlet_code, photo_type_code]):
            logger.error(f"❌ Gagal parsing folder path: {filename}")
            return False

        # Step 2: Prepare data
        serializable_faces = convert_to_json_serializable(faces)
        
        data = {
            "unit_code": unit_code,
            "photo_type_code": photo_type_code,
            "outlet_code": outlet_code,            
            "faces": safe_json_dumps(serializable_faces),
        }

        # Step 3: Network connectivity check
        logger.info(f"🔍 Checking connectivity for {filename}...")
        if not check_network_connectivity():
            logger.error(f"❌ No network connectivity for {filename}")
            return False

        # Step 4: Create robust session
        session = create_robust_session()
        url = f"{API_BASE}/faces/upload-embedding"

        # Step 5: Upload with retry mechanism
        for attempt in range(max_retries):
            try:
                logger.info(f"📤 Upload attempt {attempt + 1}/{max_retries} for {filename}")
                
                # Check server health before upload (except first attempt)
                if attempt > 0:
                    logger.info(f"🏥 Checking server health before retry...")
                    if not check_server_health(API_BASE):
                        logger.warning(f"⚠️ Server health check failed on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                            logger.info(f"⏳ Waiting {wait_time}s before next attempt...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"❌ Server unreachable after all attempts: {filename}")
                            return False

                # Attempt upload
                with open(file_path, "rb") as f:
                    files = {"file": f}
                    
                    # Progressive timeout - increase for each retry
                    current_timeout = 30 + (attempt * 15)  # 30s, 45s, 60s
                    
                    response = session.post(
                        url, 
                        data=data, 
                        files=files, 
                        timeout=current_timeout
                    )

                # Handle response
                if response.status_code == 200:
                    logger.info(f"✅ Upload berhasil: {filename}")
                    return True
                    
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # Server errors - retry
                    logger.warning(f"⚠️ Server error {response.status_code} on attempt {attempt + 1}: {filename}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Upload failed after retries - Server error {response.status_code}: {filename}")
                        return False
                        
                else:
                    # Client errors (4xx) - don't retry
                    logger.error(f"❌ Upload failed - Client error {response.status_code}: {filename}")
                    logger.error(f"Response: {response.text}")
                    return False

            except requests.exceptions.Timeout as e:
                logger.warning(f"⏰ Timeout on attempt {attempt + 1}: {filename}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"⏳ Retrying in {wait_time}s with longer timeout...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Upload timeout after all attempts: {filename}")
                    return False
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🔌 Connection error on attempt {attempt + 1}: {filename}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 4  # 4s, 8s, 16s for connection errors
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Connection error - server tidak dapat dijangkau: {filename}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"📡 Request error on attempt {attempt + 1}: {filename} - {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Request failed after all attempts: {filename}")
                    return False
                    
            except Exception as e:
                logger.warning(f"💥 Unexpected error on attempt {attempt + 1}: {filename} - {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Upload error after all attempts: {filename} - {str(e)}")
                    return False

        # Should not reach here
        logger.error(f"❌ Upload failed: {filename} - Exhausted all retry attempts")
        return False

    except Exception as e:
        logger.error(f"❌ Fatal upload error: {filename} - {str(e)}")
        return False
    
    finally:
        # Cleanup session if created
        try:
            if 'session' in locals():
                session.close()
        except:
            pass

class FaceEmbeddingWorkerSignals(QObject):
    finished = pyqtSignal(str, list, bool)  # file_path, embeddings, success
    progress = pyqtSignal(str, str)  # file_path, status
    error = pyqtSignal(str, str) 
    
class FaceEmbeddingWorker(QRunnable):
    """Optimized worker dengan progress reporting dan robust upload"""
    
    def __init__(self, file_path, allowed_paths):
        super().__init__()
        self.file_path = file_path
        self.allowed_paths = allowed_paths
        self.signals = FaceEmbeddingWorkerSignals()

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            
            self.signals.progress.emit(self.file_path, "🔍 Detecting faces...")
            
            embeddings = process_faces_in_image(self.file_path)
            
            if embeddings:
                self.signals.progress.emit(self.file_path, "📤 Uploading...")
                
                # Use enhanced upload with retry
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
            logger.error(f"Worker error for {self.file_path}: {e}")
            self.signals.error.emit(self.file_path, f"Worker error: {str(e)}")
            self.signals.finished.emit(self.file_path, [], False)