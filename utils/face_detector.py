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
from core.device_setup import device, resnet, API_BASE
from collections import defaultdict, Counter

# Shared detector instance untuk reuse
_detector_instance = None
logger = logging.getLogger(__name__)

class OptimizedYuNetDetector:
    """Optimized YuNet detector with speed improvements"""
    
    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx", conf_threshold=0.6, nms_threshold=0.3, max_size=640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_size = max_size
        self.detector = None
        self.model_warmed = False
        
        self._initialize_detector()
        self._warm_up_model()
    
    def _initialize_detector(self):
        """Initialize YuNet detector"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"❌ YuNet model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize YuNet detector
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),  # Default input size
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            
            logger.info(f"✅ YuNet detector initialized successfully")
            logger.info(f"   Model: {self.model_path}")
            logger.info(f"   Confidence threshold: {self.conf_threshold}")
            logger.info(f"   NMS threshold: {self.nms_threshold}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize YuNet detector: {e}")
            raise e
    
    def _warm_up_model(self):
        """Warm up model dengan dummy detection"""
        try:
            dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            self.detector.setInputSize((dummy_img.shape[1], dummy_img.shape[0]))
            _, faces = self.detector.detect(dummy_img)
            self.model_warmed = True
            logger.info("✅ YuNet model warmed up")
        except Exception as e:
            logger.warning(f"⚠️ Model warm up failed: {e}")
    
    def detect_with_resize(self, img):
        """Detect dengan image resizing dan coordinate scaling"""
        original_h, original_w = img.shape[:2]
        
        # Resize jika terlalu besar
        if max(original_w, original_h) > self.max_size:
            scale = self.max_size / max(original_w, original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            logger.debug(f"🔄 Resizing: {original_w}x{original_h} -> {new_w}x{new_h} (scale={scale:.3f})")
            
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Set input size for YuNet
            self.detector.setInputSize((new_w, new_h))
            _, faces = self.detector.detect(resized_img)
            
            # Scale coordinates back to original size
            if faces is not None:
                faces_scaled = []
                for face in faces:
                    # YuNet returns: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, conf]
                    x, y, w, h = face[:4]
                    conf = face[14]
                    
                    # Scale back to original size
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    
                    # Create scaled face array with confidence
                    scaled_face = [orig_x, orig_y, orig_w, orig_h, conf]
                    faces_scaled.append(scaled_face)
                    
                    logger.debug(f"Scaled bbox: ({x},{y},{w},{h}) -> ({orig_x},{orig_y},{orig_w},{orig_h})")
                
                return faces_scaled
            else:
                return None
        else:
            # No resizing needed
            self.detector.setInputSize((original_w, original_h))
            _, faces = self.detector.detect(img)
            
            if faces is not None:
                # Convert to standard format [x, y, w, h, conf]
                faces_formatted = []
                for face in faces:
                    x, y, w, h = face[:4]
                    conf = face[14]
                    faces_formatted.append([x, y, w, h, conf])
                return faces_formatted
            else:
                return None
    
    def detect(self, img):
        """Main detection method"""
        try:
            start_time = time.time()
            faces = self.detect_with_resize(img)
            detection_time = time.time() - start_time
            
            logger.info(f"🔍 YuNet detection time: {detection_time:.3f}s")
            
            if faces is None or len(faces) == 0:
                return False, None
            
            # Filter faces by confidence and validate bounding boxes
            valid_faces = []
            img_h, img_w = img.shape[:2]
            
            for face in faces:
                x, y, w, h, confidence = face
                
                # Convert to int and validate
                x, y, w, h = int(x), int(y), int(w), int(h)
                confidence = float(confidence)
                
                # Validate bbox
                if w <= 0 or h <= 0:
                    logger.debug(f"⚠️ Invalid bbox dimensions: w={w}, h={h}")
                    continue
                
                # Ensure bbox is within image bounds
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))
                
                # Check minimum face size
                if w < 20 or h < 20:
                    logger.debug(f"⚠️ Face too small: {w}x{h}")
                    continue
                
                # Filter by confidence
                if confidence >= self.conf_threshold:
                    valid_faces.append([x, y, w, h, confidence])
                    logger.debug(f"✅ Valid face: x={x}, y={y}, w={w}, h={h}, conf={confidence:.3f}")
                else:
                    logger.debug(f"⚠️ Low confidence face: {confidence:.3f} < {self.conf_threshold}")
            
            if valid_faces:
                logger.info(f"✅ Found {len(valid_faces)} valid faces")
                return True, valid_faces
            else:
                logger.info("❌ No valid faces found")
                return False, None
            
        except Exception as e:
            logger.error(f"❌ Error in YuNet detection: {e}")
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
    """Get shared YuNet detector instance"""
    global _detector_instance
    if _detector_instance is None:
        model_path = "models/face_detection_yunet_2023mar.onnx"
        _detector_instance = OptimizedYuNetDetector(
            model_path=model_path,
            conf_threshold=0.6,
            nms_threshold=0.3,
            max_size=640  # Can increase for better accuracy if needed
        )
        
        logger.info(f"🚀 YuNet face detector initialized: {model_path}")
            
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
    """Optimized face processing with YuNet detector"""
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

        # Use shared YuNet detector
        face_detector = get_shared_detector()
        success, faces = face_detector.detect(img)

        if not success or faces is None or len(faces) == 0:
            logger.warning("❌ Tidak ada wajah terdeteksi dengan YuNet.")
            return []

        logger.info(f"✅ {len(faces)} wajah terdeteksi dengan YuNet.")

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

# ===== BATCH UPLOAD UNTUK BACKEND BATCH ENDPOINT =====

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
        'User-Agent': 'FaceSync-BatchClient/1.0',
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

def batch_upload_to_backend(files_data_list, max_retries=3):
    """
    FINAL FIXED: Batch upload matching your updated backend (no file_paths parameter)
    Backend expects: files, unit_codes, photo_type_codes, outlet_codes, faces_data
    """
    
    if not files_data_list:
        logger.error("❌ No files to upload")
        return False, "No files provided"
    
    logger.info(f"🚀 Starting FINAL FIXED batch upload: {len(files_data_list)} files")
    
    try:
        # Network connectivity check
        if not check_network_connectivity():
            logger.error("❌ No network connectivity")
            return False, "No network connectivity"
        
        # Create robust session
        session = create_robust_session()
        url = f"{API_BASE}/faces/batch-upload-embedding"
        
        # Attempt upload with retry
        for attempt in range(max_retries):
            try:
                logger.info(f"📤 FINAL FIXED upload attempt {attempt + 1}/{max_retries}")
                
                # STEP 1: Prepare files
                files = []
                for i, file_data in enumerate(files_data_list):
                    file_path = file_data['file_path']
                    filename = os.path.basename(file_path)
                    
                    # Read file and add to files list
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        files.append(('files', (filename, file_content, 'image/jpeg')))
                
                # STEP 2: Prepare form data (ONLY the 4 required fields)
                data = []
                
                # Add all unit_codes (each as separate form field)
                for file_data in files_data_list:
                    data.append(('unit_codes', file_data['unit_code']))
                
                # Add all photo_type_codes
                for file_data in files_data_list:
                    data.append(('photo_type_codes', file_data['photo_type_code']))
                
                # Add all outlet_codes
                for file_data in files_data_list:
                    data.append(('outlet_codes', file_data['outlet_code']))
                
                # Add all faces_data (convert to JSON strings)
                for file_data in files_data_list:
                    serializable_faces = convert_to_json_serializable(file_data['faces'])
                    faces_json = safe_json_dumps(serializable_faces)
                    data.append(('faces_data', faces_json))
                
                # STEP 3: Validation logging
                logger.info(f"📊 FINAL FIXED format validation:")
                logger.info(f"   Files count: {len(files)}")
                logger.info(f"   Form data entries: {len(data)}")
                
                # Count fields by type
                field_counts = Counter([item[0] for item in data])
                logger.info(f"   Field counts: {dict(field_counts)}")
                
                # Validate all fields have same count as files
                expected_count = len(files_data_list)
                validation_passed = True
                
                required_fields = ['unit_codes', 'photo_type_codes', 'outlet_codes', 'faces_data']
                for field_name in required_fields:
                    count = field_counts.get(field_name, 0)
                    if count != expected_count:
                        logger.error(f"❌ Field '{field_name}' count mismatch: {count} != {expected_count}")
                        validation_passed = False
                    else:
                        logger.info(f"✅ Field '{field_name}' count correct: {count}")
                
                if not validation_passed:
                    return False, "Form data validation failed - field count mismatch"
                
                # STEP 4: Make the request
                current_timeout = 120 + (attempt * 60)
                
                logger.info(f"🌐 Making request to: {url}")
                logger.info(f"⏱️ Timeout: {current_timeout}s")
                
                response = session.post(
                    url,
                    files=files,
                    data=data,
                    timeout=current_timeout
                )
                
                logger.info(f"📡 FINAL FIXED Response status: {response.status_code}")
                
                # STEP 5: Handle response
                if response.status_code in [200, 207]:
                    try:
                        result = response.json()
                        successful = result.get('successful_uploads', 0)
                        failed = result.get('failed_uploads', 0)
                        
                        logger.info(f"✅ FINAL FIXED upload completed: {successful} successful, {failed} failed")
                        
                        # Log details if available
                        if 'successful_files' in result:
                            logger.info(f"📋 Successful files: {len(result['successful_files'])}")
                            for success_file in result['successful_files'][:3]:  # Log first 3 successes
                                logger.info(f"   ✅ {success_file.get('filename', 'unknown')}: {success_file.get('faces_count', 0)} faces")
                        
                        if 'failed_files' in result:
                            logger.info(f"📋 Failed files: {len(result['failed_files'])}")
                            for failed_file in result['failed_files'][:3]:  # Log first 3 failures
                                logger.warning(f"   ❌ {failed_file.get('filename', 'unknown')}: {failed_file.get('error', 'unknown error')}")
                        
                        return True, f"Batch upload completed: {successful}/{len(files_data_list)} successful"
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON response: {e}")
                        logger.error(f"❌ Raw response: {response.text[:500]}")
                        return False, f"Invalid JSON response: {str(e)}"
                
                elif response.status_code == 422:
                    # Validation error - log details and don't retry
                    error_text = response.text
                    logger.error(f"❌ VALIDATION ERROR 422:")
                    logger.error(f"❌ Error details: {error_text}")
                    
                    try:
                        error_json = response.json()
                        if 'detail' in error_json:
                            for error_detail in error_json['detail']:
                                logger.error(f"   Field: {error_detail.get('loc', 'unknown')}")
                                logger.error(f"   Error: {error_detail.get('msg', 'unknown')}")
                                logger.error(f"   Input: {str(error_detail.get('input', 'unknown'))[:100]}")
                    except:
                        pass
                    
                    return False, f"Validation error: {error_text[:300]}"
                
                elif response.status_code == 400:
                    # Client error (like field count mismatch) - don't retry
                    error_text = response.text
                    logger.error(f"❌ CLIENT ERROR 400:")
                    logger.error(f"❌ Error details: {error_text}")
                    return False, f"Client error: {error_text[:300]}"
                
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # Server errors - retry
                    logger.warning(f"⚠️ Server error {response.status_code} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 3
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, f"Server error {response.status_code} after {max_retries} attempts"
                
                else:
                    # Other client errors
                    error_text = response.text
                    logger.error(f"❌ Client error {response.status_code}")
                    logger.error(f"❌ Error details: {error_text[:1000]}")
                    
                    return False, f"Client error {response.status_code}: {error_text[:200]}"
            
            except requests.exceptions.Timeout as e:
                logger.error(f"⏰ Timeout error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying after timeout...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Timeout after {max_retries} attempts"
            
            except requests.exceptions.ConnectionError as e:
                logger.error(f"🔌 Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying after connection error...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Connection error after {max_retries} attempts"
            
            except Exception as e:
                logger.error(f"💥 Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Unexpected error: {str(e)}"
        
        return False, "Upload failed after all retries"
        
    except Exception as e:
        logger.error(f"❌ FINAL FIXED fatal error: {e}")
        return False, f"Fatal error: {str(e)}"

    finally:
        try:
            if 'session' in locals():
                session.close()
        except:
            pass

def validate_files_data(files_data_list):
    """
    Validate the files_data_list structure for the final backend
    """
    logger.info("🔍 Validating files data structure...")
    
    if not files_data_list:
        logger.error("❌ Empty files data list")
        return False, "No files to validate"
    
    required_fields = ['file_path', 'unit_code', 'photo_type_code', 'outlet_code', 'faces']
    errors = []
    
    for i, file_data in enumerate(files_data_list):
        # Check required fields
        for field in required_fields:
            if field not in file_data:
                errors.append(f"File {i}: Missing required field '{field}'")
            elif not file_data[field] and field != 'faces':  # faces can be empty list
                errors.append(f"File {i}: Empty value for field '{field}'")
        
        # Check file exists
        if 'file_path' in file_data:
            if not os.path.exists(file_data['file_path']):
                errors.append(f"File {i}: File not found: {file_data['file_path']}")
            elif not os.access(file_data['file_path'], os.R_OK):
                errors.append(f"File {i}: File not readable: {file_data['file_path']}")
        
        # Check faces data
        if 'faces' in file_data:
            if not isinstance(file_data['faces'], list):
                errors.append(f"File {i}: faces must be a list")
            elif len(file_data['faces']) == 0:
                logger.warning(f"⚠️ File {i}: No faces detected")
        
        # Check code fields are strings
        for code_field in ['unit_code', 'photo_type_code', 'outlet_code']:
            if code_field in file_data and not isinstance(file_data[code_field], str):
                errors.append(f"File {i}: {code_field} must be a string")
    
    if errors:
        logger.error(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors[:10]:  # Log first 10 errors
            logger.error(f"   {error}")
        return False, f"Validation failed: {len(errors)} errors"
    
    logger.info(f"✅ Validation passed for {len(files_data_list)} files")
    return True, "Validation successful"

def process_batch_faces_and_upload(files_list, allowed_paths):
    """
    FINAL VERSION: Process multiple files and upload with YuNet detector
    """
    logger.info(f"🔄 FINAL processing batch: {len(files_list)} files")
    
    # Process all files for faces first
    files_data = []
    processing_errors = []
    
    for file_path in files_list:
        try:
            filename = os.path.basename(file_path)
            logger.info(f"🔍 Processing faces: {filename}")
            
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                processing_errors.append(f"File not found: {filename}")
                continue
            
            if not os.access(file_path, os.R_OK):
                processing_errors.append(f"File not readable: {filename}")
                continue
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                processing_errors.append(f"Empty file: {filename}")
                continue
            
            # Process faces in image with YuNet
            logger.debug(f"   Detecting faces with YuNet in: {filename}")
            embeddings = process_faces_in_image(file_path)
            
            if not embeddings:
                processing_errors.append(f"No faces detected: {filename}")
                continue
            
            logger.debug(f"   Found {len(embeddings)} faces in: {filename}")
            
            # Parse path codes
            relative_path = get_relative_path(file_path, allowed_paths)
            if not relative_path:
                processing_errors.append(f"Invalid path: {filename}")
                continue
            
            unit_code, photo_type_code, outlet_code = parse_codes_from_relative_path(
                relative_path, allowed_paths[0]
            )
            
            if not all([unit_code, outlet_code, photo_type_code]):
                processing_errors.append(f"Path parsing failed: {filename} (unit={unit_code}, outlet={outlet_code}, photo_type={photo_type_code})")
                continue
            
            # Validate codes are strings
            if not all(isinstance(code, str) for code in [unit_code, outlet_code, photo_type_code]):
                processing_errors.append(f"Invalid code types: {filename}")
                continue
            
            # Add to batch data
            files_data.append({
                'file_path': file_path,
                'unit_code': unit_code,
                'photo_type_code': photo_type_code,  
                'outlet_code': outlet_code,
                'faces': embeddings
            })
            
            logger.info(f"✅ Processed: {filename} ({len(embeddings)} faces) - {unit_code}/{outlet_code}/{photo_type_code}")
            
        except Exception as e:
            error_msg = f"Processing error for {os.path.basename(file_path)}: {str(e)}"
            processing_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    # Log processing summary
    logger.info(f"📊 Processing complete: {len(files_data)} ready for upload, {len(processing_errors)} errors")
    
    if processing_errors:
        logger.warning(f"⚠️ Processing errors ({len(processing_errors)}):")
        for error in processing_errors[:5]:  # Log first 5 errors
            logger.warning(f"   {error}")
        if len(processing_errors) > 5:
            logger.warning(f"   ... and {len(processing_errors) - 5} more errors")
    
    if not files_data:
        return False, "No files ready for upload"
    
    # Validate files data structure
    is_valid, validation_message = validate_files_data(files_data)
    if not is_valid:
        return False, f"Validation failed: {validation_message}"
    
    # Upload batch using the final fixed version
    logger.info("🚀 Starting upload with FINAL FIXED version...")
    success, message = batch_upload_to_backend(files_data)
    
    if success:
        logger.info(f"✅ FINAL FIXED upload successful: {message}")
    else:
        logger.error(f"❌ FINAL FIXED upload failed: {message}")
    
    return success, message

def debug_final_structure(files_data_list):
    """
    Debug function to show exactly what will be sent to the final backend
    """
    logger.info("🧪 DEBUG: Final structure validation")
    
    if not files_data_list:
        logger.error("No test data provided")
        return
    
    # Show what we'll send
    files_count = len(files_data_list)
    logger.info(f"Files to upload: {files_count}")
    
    # Check each required field
    required_fields = ['unit_code', 'photo_type_code', 'outlet_code', 'faces']
    
    for field in required_fields:
        values = [fd.get(field, 'MISSING') for fd in files_data_list]
        logger.info(f"{field}: {len(values)} values")
        for i, value in enumerate(values[:3]):  # Show first 3
            if field == 'faces':
                logger.info(f"  [{i}] {type(value).__name__} with {len(value) if isinstance(value, list) else 'N/A'} items")
            else:
                logger.info(f"  [{i}] {value}")
        if len(values) > 3:
            logger.info(f"  ... and {len(values) - 3} more")
    
    # Validate all fields have same count
    logger.info("✅ All field counts match files count" if all(
        len([fd.get(field) for fd in files_data_list]) == files_count 
        for field in required_fields
    ) else "❌ Field count mismatch detected")

class BatchFaceEmbeddingWorkerSignals(QObject):
    finished = pyqtSignal(str, bool, str)  # result_summary, success, message
    progress = pyqtSignal(str, str)  # current_file, status
    error = pyqtSignal(str, str)  # file_path, error_message
    batch_completed = pyqtSignal(int, int)  # successful_count, failed_count

class BatchFaceEmbeddingWorker(QRunnable):
    """NEW: Batch worker yang menggunakan YuNet detector dan backend batch endpoint"""
    
    def __init__(self, files_list, allowed_paths):
        super().__init__()
        self.files_list = files_list
        self.allowed_paths = allowed_paths
        self.signals = BatchFaceEmbeddingWorkerSignals()

    def run(self):
        try:
            batch_size = len(self.files_list)
            self.signals.progress.emit("batch", f"🔄 Processing batch of {batch_size} files with YuNet...")
            
            # Process and upload batch
            success, message = process_batch_faces_and_upload(self.files_list, self.allowed_paths)
            
            if success:
                self.signals.progress.emit("batch", "✅ Batch upload completed")
                self.signals.finished.emit(f"Batch successful: {batch_size} files", True, message)
                self.signals.batch_completed.emit(batch_size, 0)  # All successful for now
            else:
                self.signals.progress.emit("batch", "❌ Batch upload failed")
                self.signals.finished.emit(f"Batch failed: {batch_size} files", False, message)
                self.signals.batch_completed.emit(0, batch_size)  # All failed for now
                
        except Exception as e:
            error_message = f"Batch worker error: {str(e)}"
            logger.error(f"❌ {error_message}")
            self.signals.error.emit("batch", error_message)
            self.signals.finished.emit("Batch error", False, error_message)
            self.signals.batch_completed.emit(0, len(self.files_list))

# Legacy single upload function (kept for compatibility)
def upload_embedding_to_backend(file_path, faces, allowed_paths, max_retries=3):
    """
    LEGACY: Single file upload - kept for backward compatibility
    For new implementations, use batch_upload_to_backend instead
    """
    
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

        # Step 2: Use batch upload with single file for consistency
        files_data = [{
            'file_path': file_path,
            'unit_code': unit_code,
            'photo_type_code': photo_type_code,
            'outlet_code': outlet_code,
            'faces': faces
        }]
        
        success, message = batch_upload_to_backend(files_data, max_retries)
        
        if success:
            logger.info(f"✅ Single upload successful: {filename}")
        else:
            logger.error(f"❌ Single upload failed: {filename} - {message}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Single upload error: {filename} - {str(e)}")
        return False

class FaceEmbeddingWorkerSignals(QObject):
    finished = pyqtSignal(str, list, bool)  # file_path, embeddings, success
    progress = pyqtSignal(str, str)  # file_path, status
    error = pyqtSignal(str, str) 
    
class FaceEmbeddingWorker(QRunnable):
    """LEGACY: Single file worker with YuNet - kept for backward compatibility"""
    
    def __init__(self, file_path, allowed_paths):
        super().__init__()
        self.file_path = file_path
        self.allowed_paths = allowed_paths
        self.signals = FaceEmbeddingWorkerSignals()

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            
            self.signals.progress.emit(self.file_path, "🔍 Detecting faces with YuNet...")
            
            embeddings = process_faces_in_image(self.file_path)
            
            if embeddings:
                self.signals.progress.emit(self.file_path, "📤 Uploading...")
                
                # Use single upload (which internally uses batch with 1 file)
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