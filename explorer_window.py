import sys
import os
from dotenv import load_dotenv
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileSystemModel, QTreeView, QListView,
    QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QAction,
    QInputDialog, QMessageBox, QAbstractItemView, QDialog, QFormLayout,
    QDialogButtonBox, QTabWidget, QGroupBox, QCheckBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QIcon, QDrag, QClipboard
from PyQt5.QtCore import Qt, QDir, QMimeData, QUrl, QThread, pyqtSignal, QTimer
from watcher import start_watcher, stop_watcher

from admin_setup_dialogs import AdminSetupDialog
from admin_login import AdminLoginDialog
from admin_setting import AdminSettingsDialog
from config_manager import ConfigManager
from features import DragDropListWidget
import logging
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import torch
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject
import requests
import json
import onnxruntime as ort
from retinaface import RetinaFace
import logging
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor

# Ambil dari environment
load_dotenv()
API_BASE = os.getenv("BASE_URL")

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Move to GPU
logger = logging.getLogger(__name__)

# Print GPU info saat startup
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🚀 GPU detected: {gpu_name}")
    logger.info(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    logger.info("💻 Using CPU processing")

# Shared detector instance untuk reuse
_detector_instance = None

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
        if len(parts) < 3:
            logger.warning(f"Path tidak lengkap: {relative_path}")
            return None, None, None

        unit_folder = os.path.basename(allowed_path.rstrip(os.sep))
        outlet_folder = parts[0]
        photo_type_folder = parts[1]

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

        unit_code, outlet_code, photo_type_code = parse_codes_from_relative_path(
            relative_path, allowed_paths[0]
        )
        if not all([unit_code, outlet_code, photo_type_code]):
            logger.error("Gagal parsing folder path.")
            return False

        # Ensure faces are JSON serializable
        serializable_faces = convert_to_json_serializable(faces)

        data = {
            "unit_code": unit_code,
            "outlet_code": outlet_code,
            "photo_type_code": photo_type_code,
            "file_path": file_path,
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

class WatcherThread(QThread):
    """Optimized watcher thread"""
    new_file_signal = pyqtSignal(str)
    deleted_file_signal = pyqtSignal(str)

    def __init__(self, folder_path, recursive=True):
        super().__init__()
        self.folder_path = folder_path
        self.recursive = recursive
        self.observer = None

    def run(self):
        self.observer = start_watcher(
            self.folder_path,
            self.handle_new_file,
            self.handle_deleted_file,
            recursive=self.recursive
        )
        self.exec_()

    def handle_new_file(self, file_path):
        self.new_file_signal.emit(file_path)

    def handle_deleted_file(self, file_path):
        self.deleted_file_signal.emit(file_path)

    def stop(self):
        if self.observer:
            stop_watcher(self.observer)
        self.quit()
        self.wait()

class ExplorerWindow(QMainWindow):
    """Optimized main window dengan performance improvements"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # Limit concurrent processing
        self.config_manager = config_manager
        self.setWindowTitle("Auto Uploader - Ownize Explorer")
        self.setGeometry(100, 100, 1200, 700)
        self.watcher_thread = None

        # Initialize UI
        self._init_ui()
        self._setup_connections()
        
        # Status tracking
        self.embedding_in_progress = 0
        self.processing_files = {}  # Track files being processed
        
        # Navigation
        self.path_history = []
        self.current_path = ""
        self.allowed_paths = self.config_manager.config.get("allowed_paths", [])
        
        # Auto-load initial path
        self._load_initial_path()

    def _init_ui(self):
        """Initialize UI components"""
        # Top controls
        self.path_display = QLabel()
        self.path_display.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        self.path_input = QLineEdit()
        self.path_input.setVisible(False)        
        self.admin_button = QPushButton("Admin Settings")
        self.back_button = QPushButton("← Back")
        self.back_button.setEnabled(False)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(QLabel("📁"))
        top_layout.addWidget(self.path_display)
        top_layout.addStretch()        
        top_layout.addWidget(self.admin_button)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search files...")

        # File list dengan improved settings
        self.file_list = DragDropListWidget(self)
        self.file_list.setViewMode(QListView.IconMode)
        self.file_list.setIconSize(QPixmap(100, 100).size())
        self.file_list.setResizeMode(QListView.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setWordWrap(True)
        self.file_list.setGridSize(QPixmap(140, 140).size())

        # Progress bar for processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.search_input)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.file_list)
        main_layout.addWidget(QLabel("Logs:"))
        main_layout.addWidget(self.log_text)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self.embedding_label = QLabel()
        self.status_bar.addPermanentWidget(self.embedding_label)

    def _setup_connections(self):
        """Setup signal connections"""
        
        self.admin_button.clicked.connect(self.show_admin_settings)
        self.back_button.clicked.connect(self.go_back)
        self.file_list.itemDoubleClicked.connect(self.open_file)
        self.search_input.textChanged.connect(self.filter_file_list)
        self.path_input.textChanged.connect(self.on_path_changed)

    def _load_initial_path(self):
        """Load initial path if available"""
        if self.allowed_paths:
            initial_path = self.allowed_paths[0]
            self.set_current_path(initial_path)
            self.load_files(initial_path)
            self.start_monitoring(initial_path)

    def update_embedding_status(self):
        """Update embedding status display"""
        if self.embedding_in_progress > 0:
            self.embedding_label.setText(f"🧠 Processing: {self.embedding_in_progress} files")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
        else:
            self.embedding_label.setText("")
            self.progress_bar.setVisible(False)
            # Clear progress label juga ketika semua selesai
            if self.embedding_in_progress == 0:
                self.progress_label.setText("")

    def _clear_progress_if_done(self):
        """Clear progress label jika tidak ada yang sedang diproses"""
        if self.embedding_in_progress == 0:
            self.progress_label.setText("")

    def update_progress_label(self, file_path, status):
        """Update progress label"""
        filename = os.path.basename(file_path)
        self.progress_label.setText(f"{status} {filename}")
        
        # Auto-clear setelah delay untuk status upload
        if "📤" in status:  # Upload status
            QTimer.singleShot(2000, self._clear_progress_if_done)  # Clear setelah 2 detik

    def filter_file_list(self, text):
        """Filter file list based on search text"""
        text = text.lower()
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            filename = item.data(Qt.UserRole).lower()
            item.setHidden(text not in filename)

    def set_current_path(self, path):
        """Set current path and update display"""
        self.current_path = path
        self.path_input.setText(path)
        display_name = os.path.basename(path) if path else ""
        if not display_name:
            display_name = path
        self.path_display.setText(display_name)

    def smart_truncate_filename(self, filename, max_chars=16):
        """Smart truncate filename preserving extension"""
        if len(filename) <= max_chars:
            return filename
        
        if '.' in filename:
            name_part, ext = os.path.splitext(filename)
            available_chars = max_chars - len(ext) - 3
            if available_chars > 3:
                return name_part[:available_chars] + "..." + ext
            else:
                return filename[:max_chars-3] + "..."
        else:
            return filename[:max_chars-3] + "..."

    def log_with_timestamp(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def browse_folder(self):
        """Browse and select folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            if not self.config_manager.is_path_allowed(folder):
                QMessageBox.warning(self, "Access Denied", 
                                  "Path ini tidak diizinkan!\nSilakan hubungi admin.")
                return
            
            if self.current_path and self.current_path != folder:
                self.path_history.append(self.current_path)
                self.back_button.setEnabled(True)
            
            self.set_current_path(folder)
            self.load_files(folder)

    def go_back(self):
        """Navigate back to previous folder"""
        if self.path_history:
            previous_path = self.path_history.pop()
            self.set_current_path(previous_path)
            self.load_files(previous_path)
            self.back_button.setEnabled(len(self.path_history) > 0)
            self.log_with_timestamp(f"⬅️ Back to: {os.path.basename(previous_path)}")
            self.on_path_changed(previous_path)

    def load_files(self, folder_path):
        """Load files from folder with optimization"""
        if not self.config_manager.is_path_allowed(folder_path):
            self.log_with_timestamp(f"❌ Access denied: {folder_path}")
            return
        
        self.file_list.clear()
        if not os.path.exists(folder_path):
            return
            
        try:
            image_exts = ['.png', '.jpg', '.jpeg']
            items = os.listdir(folder_path)
            
            # Separate and sort items
            folders = sorted([item for item in items 
                            if os.path.isdir(os.path.join(folder_path, item))])
            image_files = sorted([item for item in items 
                                if os.path.splitext(item)[1].lower() in image_exts])
            
            # Add folders first
            for folder_name in folders:
                self._add_folder_item(folder_name, folder_path)
            
            # Add image files
            for filename in image_files:
                self._add_image_item(filename, folder_path)
                
        except Exception as e:
            self.log_with_timestamp(f"❌ Error loading folder: {str(e)}")

    def _add_folder_item(self, folder_name, folder_path):
        """Add folder item to list"""
        item = QListWidgetItem()
        display_name = self.smart_truncate_filename(folder_name, max_chars=16)
        item.setText(display_name)
        item.setData(Qt.UserRole, folder_name)
        item.setData(Qt.UserRole + 1, "folder")
        item.setIcon(self.style().standardIcon(self.style().SP_DirIcon))
        
        full_path = os.path.join(folder_path, folder_name)
        item.setToolTip(f"Folder: {folder_name}\nPath: {full_path}")
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        
        self.file_list.addItem(item)

    def _add_image_item(self, filename, folder_path):
        """Add image item to list"""
        item = QListWidgetItem()
        display_name = self.smart_truncate_filename(filename, max_chars=16)
        item.setText(display_name)
        item.setData(Qt.UserRole, filename)
        item.setData(Qt.UserRole + 1, "image")
        
        full_path = os.path.join(folder_path, filename)
        
        # Load thumbnail asynchronously in production
        pixmap = QPixmap(full_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(scaled_pixmap))
        else:
            item.setIcon(self.style().standardIcon(self.style().SP_FileIcon))
        
        # File info tooltip
        try:
            file_info = os.stat(full_path)
            size_mb = file_info.st_size / (1024 * 1024)
            item.setToolTip(f"Image: {filename}\nSize: {size_mb:.2f} MB")
        except:
            item.setToolTip(f"Image: {filename}")
        
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.file_list.addItem(item)

    def open_file(self, item):
        """Open file or navigate to folder"""
        item_name = self.file_list.get_actual_filename(item)
        item_path = os.path.join(self.current_path, item_name)
        item_type = item.data(Qt.UserRole + 1)
        
        if item_type == "folder":
            if self.config_manager.is_path_allowed(item_path):
                if self.current_path != item_path and os.path.exists(self.current_path):
                    self.path_history.append(self.current_path)
                    self.back_button.setEnabled(True)
                
                self.set_current_path(item_path)
                self.load_files(item_path)
                self.on_path_changed(item_path)
            else:
                QMessageBox.warning(self, "Access Denied", 
                                  f"Folder '{item_name}' tidak dapat diakses!")
        else:
            # Open image with system default
            try:
                if sys.platform == "win32":
                    os.startfile(item_path)
                elif sys.platform == "darwin":
                    os.system(f"open '{item_path}'")
                else:
                    os.system(f"xdg-open '{item_path}'")
                
                self.log_with_timestamp(f"🔍 Opened: {item_name}")
            except Exception as e:
                self.log_with_timestamp(f"❌ Error opening {item_name}: {str(e)}")

    def on_path_changed(self, new_path):
        """Handle path change"""
        if new_path and os.path.isdir(new_path) and self.config_manager.is_path_allowed(new_path):
            if self.watcher_thread:
                self.stop_monitoring()
            self.start_monitoring(new_path)

    def start_monitoring(self, folder=None):
        """Start folder monitoring"""
        if folder is None:
            folder = self.current_path
            
        if not os.path.isdir(folder) or not self.config_manager.is_path_allowed(folder):
            return
        
        if (self.watcher_thread and 
            hasattr(self.watcher_thread, 'folder_path') and 
            self.watcher_thread.folder_path == folder):
            return
        
        self.watcher_thread = WatcherThread(folder, recursive=True)
        self.watcher_thread.new_file_signal.connect(self.on_new_file_detected)
        self.watcher_thread.deleted_file_signal.connect(self.on_file_deleted)
        self.watcher_thread.start()
        
        self.status_bar.showMessage(f"🔄 Monitoring: {os.path.basename(folder)}")

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.watcher_thread:
            self.watcher_thread.stop()
            self.watcher_thread = None
            self.status_bar.showMessage("Ready")

    def on_new_file_detected(self, file_path):
        """Handle new file detection"""
        filename = os.path.basename(file_path)
        
        image_exts = ['.png', '.jpg', '.jpeg']
        if os.path.splitext(filename)[1].lower() not in image_exts:
            return
            
        self.log_with_timestamp(f"🆕 New image: {filename}")
        
        # Add to file list if in current folder
        file_dir = os.path.dirname(file_path)
        if file_dir == self.current_path:
            self._add_image_item(filename, file_dir)
        
        # Process faces
        if file_path not in self.processing_files:
            self.processing_files[file_path] = True
            worker = FaceEmbeddingWorker(file_path, self.allowed_paths)
            worker.signals.finished.connect(self.on_embedding_finished)
            worker.signals.progress.connect(self.update_progress_label)
            
            self.embedding_in_progress += 1
            self.update_embedding_status()
            self.threadpool.start(worker)

    def on_embedding_finished(self, file_path, embeddings, success):
        """Handle embedding completion"""
        self.embedding_in_progress = max(0, self.embedding_in_progress - 1)
        self.processing_files.pop(file_path, None)
        self.update_embedding_status()
        
        # Clear progress label ketika tidak ada lagi yang diproses
        if self.embedding_in_progress == 0:
            self.progress_label.setText("")
        
        filename = os.path.basename(file_path)
        if success and embeddings:
            self.log_with_timestamp(f"✅ Processed: {filename} ({len(embeddings)} faces)")
        else:
            self.log_with_timestamp(f"❌ Failed: {filename}")

    def on_file_deleted(self, file_path):
        """Handle file deletion"""
        filename = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, self.current_path)
        self.log_with_timestamp(f"🗑️ Deleted: {relative_path}")
        
        # Remove from current view if applicable
        file_dir = os.path.dirname(file_path)
        if file_dir == self.current_path:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.UserRole) == filename:
                    self.file_list.takeItem(i)
                    break

    def show_admin_settings(self):
        """Show admin settings"""
        login_dialog = AdminLoginDialog(self.config_manager, self)
        if login_dialog.exec_() == QDialog.Accepted:
            settings_dialog = AdminSettingsDialog(self.config_manager, self)
            settings_dialog.exec_()
        else:
            QMessageBox.information(self, "Info", "Login diperlukan untuk admin settings.")

    def closeEvent(self, event):
        """Handle application close"""
        if self.watcher_thread:
            self.stop_monitoring()
        
        # Wait for running workers to complete
        self.threadpool.waitForDone(3000)  # 3 second timeout
        event.accept()
