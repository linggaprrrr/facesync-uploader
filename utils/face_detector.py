import logging
import cv2
import os
import numpy as np
import torch
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject
import requests
import json
from retinaface import RetinaFace
import logging
import time
from core.device_setup import device, resnet, API_BASE

# Shared detector instance untuk reuse
_detector_instance = None
logger = logging.getLogger(__name__)

class OptimizedRetinaFaceDetector:
    """Optimized RetinaFace detector dengan speed improvements"""
    
    def __init__(self, device='cpu', conf_threshold=0.6, nms_threshold=0.4, max_size=640):
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_size = max_size
        self.model_warmed = False
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

def process_faces_in_image(file_path, original_shape=None, pad=None, scale=None):
    """Optimized face processing dengan error handling yang lebih baik"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            logger.warning(f"❌ Gagal membaca gambar: {file_path}")
            return []

        h, w = img.shape[:2]
        logger.info(f"📸 Processing image: {w}x{h}")

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
                
                # Validasi koordinat
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w_box, w), min(y + h_box, h)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"⚠️ Invalid bbox untuk wajah {i}")
                    continue
                    
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Optimized preprocessing
                face_crop_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                face_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
                
                # Tensor conversion optimization dengan GPU support
                face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
                face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                face_tensor = face_tensor.unsqueeze(0).to(device)  # Move to GPU

                with torch.no_grad():
                    embedding_tensor = resnet(face_tensor).squeeze()
                    embedding = embedding_tensor.cpu().numpy().tolist()  # Move back to CPU for JSON

                # Bbox calculation
                if original_shape and pad and scale:
                    bbox_dict = {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)}
                    original_bbox = reverse_letterbox(
                        bbox=bbox_dict,
                        original_shape=original_shape,
                        resized_shape=img.shape[:2],
                        pad=pad,
                        scale=scale
                    )
                    original_bbox = convert_to_json_serializable(original_bbox)
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

def upload_embedding_to_backend(file_path, faces, allowed_paths):
    """Upload embedding dengan better error handling"""
    try:
        relative_path = get_relative_path(file_path, allowed_paths)
        if not relative_path:
            logger.error("File path tidak termasuk folder yang diizinkan.")
            return False

        unit_code, photo_type_code, outlet_code = parse_codes_from_relative_path(
            relative_path, allowed_paths[0]
        )

        
        if not all([unit_code, outlet_code, photo_type_code]):
            logger.error("Gagal parsing folder path.")
            return False

        # Ensure faces are JSON serializable
        serializable_faces = convert_to_json_serializable(faces)
        
        data = {
            "unit_code": unit_code,
            "photo_type_code": photo_type_code,
            "outlet_code": outlet_code,            
            "faces": safe_json_dumps(serializable_faces),
        }

        with open(file_path, "rb") as f:
            files = {"file": f}
            url = f"{API_BASE}/faces/upload-embedding"
            
            # Add timeout untuk prevent hanging
            response = requests.post(url, data=data, files=files, timeout=30)

        if response.status_code == 200:
            logger.info("✅ Upload berhasil.")
            return True
        else:
            logger.error(f"❌ Upload gagal: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error("❌ Upload timeout - server tidak merespons")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Connection error - server tidak dapat dijangkau")
        return False
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        return False

class FaceEmbeddingWorkerSignals(QObject):
    finished = pyqtSignal(str, list, bool)  # file_path, embeddings, success
    progress = pyqtSignal(str, str)  # file_path, status

class FaceEmbeddingWorker(QRunnable):
    """Optimized worker dengan progress reporting"""
    
    def __init__(self, file_path, allowed_paths):
        super().__init__()
        self.file_path = file_path
        self.allowed_paths = allowed_paths
        self.signals = FaceEmbeddingWorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit(self.file_path, "🔍 Detecting faces...")
            
            embeddings = process_faces_in_image(self.file_path)
            
            if embeddings:
                self.signals.progress.emit(self.file_path, "📤 Uploading...")
                success = upload_embedding_to_backend(
                    self.file_path, embeddings, self.allowed_paths
                )
                self.signals.finished.emit(self.file_path, embeddings, success)
            else:
                self.signals.finished.emit(self.file_path, [], False)
                
        except Exception as e:
            logger.error(f"Worker error for {self.file_path}: {e}")
            self.signals.finished.emit(self.file_path, [], False)