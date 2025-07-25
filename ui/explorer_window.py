import sys
import os
import time
import threading
import hashlib
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from PyQt5.QtWidgets import (
    QMainWindow, QListView,
    QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QListWidgetItem, QMessageBox, QAbstractItemView, QDialog, QProgressBar
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from core.watcher import start_watcher, stop_watcher
from ui.admin_login import AdminLoginDialog
from ui.admin_setting import AdminSettingsDialog
from ui.features import DragDropListWidget
from PyQt5.QtCore import QThreadPool
import cv2
import numpy as np
from utils.face_detector import FaceEmbeddingWorker

logger = logging.getLogger(__name__)

class FileState(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class FileInfo:
    path: str
    size: int
    modified_time: float
    hash: Optional[str] = None
    state: FileState = FileState.PENDING
    retry_count: int = 0
    last_attempt: float = 0

class ProductionFileIntegrityHandler:
    """Production-ready file integrity handler"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 initial_delay: float = 0.2,
                 max_delay: float = 2.0,
                 stability_check_duration: float = 0.5,
                 file_timeout: float = 15.0,
                 cleanup_interval: float = 300.0):
        
        # Configuration
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.stability_check_duration = stability_check_duration
        self.file_timeout = file_timeout
        self.cleanup_interval = cleanup_interval
        
        # State management
        self.file_registry: Dict[str, FileInfo] = {}
        self.processing_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_timer = None
        
        # Callbacks
        self.on_file_ready: Optional[Callable[[str], None]] = None
        self.on_file_failed: Optional[Callable[[str, str], None]] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup timer
        self._start_cleanup_timer()
    
    def register_file(self, file_path: str) -> bool:
        """Register a new file for processing"""
        file_path = os.path.abspath(file_path)
        
        with self._lock:
            # Skip if already processing or recently failed
            if (file_path in self.processing_files or 
                file_path in self.failed_files):
                return False
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                return False
            
            try:
                stat = os.stat(file_path)
                file_info = FileInfo(
                    path=file_path,
                    size=stat.st_size,
                    modified_time=stat.st_mtime,
                    state=FileState.PENDING
                )
                
                self.file_registry[file_path] = file_info
                self.logger.info(f"Registered file: {os.path.basename(file_path)}")
                
                # Start processing in background thread
                threading.Thread(
                    target=self._process_file_async,
                    args=(file_path,),
                    daemon=True,
                    name=f"FileProcessor-{os.path.basename(file_path)}"
                ).start()
                
                return True
                
            except (OSError, IOError) as e:
                self.logger.warning(f"Failed to register {file_path}: {e}")
                return False
    
    def _process_file_async(self, file_path: str):
        """Process file asynchronously with full error handling"""
        filename = os.path.basename(file_path)
        
        try:
            with self._lock:
                if file_path not in self.file_registry:
                    return
                self.processing_files.add(file_path)
                self.file_registry[file_path].state = FileState.PROCESSING
            
            # Wait for file to be ready
            if self._wait_for_file_ready(file_path):
                # File is ready, notify callback
                if self.on_file_ready:
                    try:
                        self.on_file_ready(file_path)
                        with self._lock:
                            self.file_registry[file_path].state = FileState.COMPLETED
                        self.logger.info(f"Successfully processed: {filename}")
                    except Exception as e:
                        self._handle_file_failure(file_path, f"Callback error: {e}")
                else:
                    with self._lock:
                        self.file_registry[file_path].state = FileState.COMPLETED
            else:
                self._handle_file_failure(file_path, "File readiness timeout")
                
        except Exception as e:
            self._handle_file_failure(file_path, f"Processing error: {e}")
        finally:
            with self._lock:
                self.processing_files.discard(file_path)
    
    def _wait_for_file_ready(self, file_path: str) -> bool:
        """Comprehensive file readiness check"""
        start_time = time.time()
        
        while time.time() - start_time < self.file_timeout:
            try:
                file_info = self.file_registry.get(file_path)
                if not file_info:
                    return False
                
                # Check if we should retry (exponential backoff)
                if not self._should_retry(file_info):
                    time.sleep(0.1)
                    continue
                
                # Update retry info
                with self._lock:
                    file_info.retry_count += 1
                    file_info.last_attempt = time.time()
                
                # Comprehensive readiness check
                if self._comprehensive_file_check(file_path):
                    return True
                
                # Calculate next retry delay (exponential backoff)
                delay = min(
                    self.initial_delay * (2 ** file_info.retry_count),
                    self.max_delay
                )
                time.sleep(delay)
                
            except Exception as e:
                self.logger.warning(f"Error in readiness check for {file_path}: {e}")
                time.sleep(self.initial_delay)
        
        return False
    
    def _should_retry(self, file_info: FileInfo) -> bool:
        """Check if we should attempt another retry"""
        if file_info.retry_count >= self.max_retries:
            return False
        
        # Respect exponential backoff timing
        if file_info.last_attempt > 0:
            min_wait = self.initial_delay * (2 ** file_info.retry_count)
            elapsed = time.time() - file_info.last_attempt
            if elapsed < min_wait:
                return False
        
        return True
    
    def _comprehensive_file_check(self, file_path: str) -> bool:
        """Multi-layered file integrity verification"""
        try:
            # 1. Basic existence and permissions
            if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                return False
            
            # 2. File size stability check
            if not self._is_file_size_stable(file_path):
                return False
            
            # 3. File lock check (platform-specific)
            if not self._can_access_exclusively(file_path):
                return False
            
            # 4. Image integrity verification
            if not self._verify_image_integrity(file_path):
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"File check failed for {file_path}: {e}")
            return False
    
    def _is_file_size_stable(self, file_path: str) -> bool:
        """Check if file size remains stable"""
        try:
            initial_size = os.path.getsize(file_path)
            if initial_size == 0:
                return False
            
            time.sleep(self.stability_check_duration)
            
            final_size = os.path.getsize(file_path)
            return initial_size == final_size
            
        except (OSError, IOError):
            return False
    
    def _can_access_exclusively(self, file_path: str) -> bool:
        """Platform-specific exclusive access check"""
        try:
            if os.name == 'nt':  # Windows
                return self._windows_lock_check(file_path)
            else:  # Unix-like
                return self._unix_lock_check(file_path)
        except Exception:
            return False
    
    def _windows_lock_check(self, file_path: str) -> bool:
        """Windows-specific file lock check"""
        try:
            import msvcrt
            with open(file_path, 'rb') as f:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            return True
        except (ImportError, OSError, IOError):
            return False
    
    def _unix_lock_check(self, file_path: str) -> bool:
        """Unix-specific file lock check"""
        try:
            import fcntl
            with open(file_path, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except (ImportError, OSError, IOError):
            return False
    
    def _verify_image_integrity(self, file_path: str) -> bool:
        """Verify image can be read and is valid"""
        try:
            # Try reading with cv2
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False
            
            # Verify it's a valid numpy array with data
            if not isinstance(img, np.ndarray) or img.size == 0:
                return False
            
            # Check image dimensions are reasonable
            if len(img.shape) < 2 or any(dim <= 0 for dim in img.shape[:2]):
                return False
            
            # Try to access image data (will fail if corrupted)
            _ = img[0, 0]
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Image integrity check failed: {e}")
            return False
    
    def _handle_file_failure(self, file_path: str, reason: str):
        """Handle file processing failure"""
        filename = os.path.basename(file_path)
        
        with self._lock:
            if file_path in self.file_registry:
                self.file_registry[file_path].state = FileState.FAILED
            self.failed_files.add(file_path)
        
        self.logger.error(f"File processing failed for {filename}: {reason}")
        
        if self.on_file_failed:
            try:
                self.on_file_failed(file_path, reason)
            except Exception as e:
                self.logger.error(f"Error in failure callback: {e}")
    
    def _start_cleanup_timer(self):
        """Start periodic cleanup of old file records"""
        def cleanup():
            self._cleanup_old_records()
            # Schedule next cleanup
            self._cleanup_timer = threading.Timer(
                self.cleanup_interval, 
                cleanup
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
        
        cleanup()
    
    def _cleanup_old_records(self):
        """Clean up old file records to prevent memory leaks"""
        current_time = time.time()
        cutoff_time = current_time - self.cleanup_interval
        
        with self._lock:
            # Clean up completed and failed files older than cleanup interval
            paths_to_remove = []
            
            for file_path, file_info in self.file_registry.items():
                if (file_info.state in [FileState.COMPLETED, FileState.FAILED] and
                    file_info.last_attempt < cutoff_time):
                    paths_to_remove.append(file_path)
            
            for path in paths_to_remove:
                del self.file_registry[path]
                self.failed_files.discard(path)
        
        if paths_to_remove:
            self.logger.info(f"Cleaned up {len(paths_to_remove)} old file records")
    
    def get_status(self) -> Dict:
        """Get current status of file handler"""
        with self._lock:
            return {
                'total_files': len(self.file_registry),
                'processing': len(self.processing_files),
                'completed': len([f for f in self.file_registry.values() 
                                if f.state == FileState.COMPLETED]),
                'failed': len(self.failed_files),
                'pending': len([f for f in self.file_registry.values() 
                              if f.state == FileState.PENDING])
            }
    
    def reset_failed_file(self, file_path: str) -> bool:
        """Reset a failed file for retry"""
        with self._lock:
            if file_path in self.failed_files:
                self.failed_files.remove(file_path)
                if file_path in self.file_registry:
                    file_info = self.file_registry[file_path]
                    file_info.state = FileState.PENDING
                    file_info.retry_count = 0
                    file_info.last_attempt = 0
                return True
        return False
    
    def shutdown(self):
        """Clean shutdown of the handler"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        with self._lock:
            self.processing_files.clear()
            self.file_registry.clear()
            self.failed_files.clear()

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
    """Enhanced Explorer Window with Production File Integrity Handler"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # Limit concurrent processing
        self.config_manager = config_manager
        self.setWindowTitle("Auto Uploader - Ownize Explorer")
        self.setGeometry(100, 100, 1200, 700)
        self.watcher_thread = None

        # 🚀 NEW: Initialize production file integrity handler
        self.file_handler = ProductionFileIntegrityHandler(
            max_retries=3,
            initial_delay=0.2,
            stability_check_duration=0.5,
            file_timeout=15.0
        )
        
        # Set up callbacks for file handler
        self.file_handler.on_file_ready = self._process_ready_file
        self.file_handler.on_file_failed = self._handle_failed_file

        # Initialize UI
        self._init_ui()
        self._setup_connections()
        
        # Status tracking
        self.embedding_in_progress = 0
        self.processing_files = {}  # Keep for backward compatibility
        
        # Navigation
        self.path_history = []
        self.current_path = ""
        self.allowed_paths = self.config_manager.config.get("allowed_paths", [])
        
        # Supported image extensions
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tga'}
        
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

        # 🆕 NEW: Add retry failed files button
        self.retry_button = QPushButton("🔄 Retry Failed")
        self.retry_button.clicked.connect(self.retry_failed_files)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(QLabel("📁"))
        top_layout.addWidget(self.path_display)
        top_layout.addStretch()
        top_layout.addWidget(self.retry_button)  # 🆕 NEW
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
        """Update embedding status display with enhanced info"""
        # Get file handler status
        handler_status = self.file_handler.get_status()
        
        if self.embedding_in_progress > 0 or handler_status['processing'] > 0:
            total_processing = self.embedding_in_progress + handler_status['processing']
            status_text = f"🧠 Processing: {total_processing} files"
            
            # Add failed count if any
            if handler_status['failed'] > 0:
                status_text += f" | ❌ Failed: {handler_status['failed']}"
            
            self.embedding_label.setText(status_text)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
        else:
            failed_count = handler_status['failed']
            if failed_count > 0:
                self.embedding_label.setText(f"❌ {failed_count} failed files (click Retry)")
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
            items = os.listdir(folder_path)
            
            # Separate and sort items
            folders = sorted([item for item in items 
                            if os.path.isdir(os.path.join(folder_path, item))])
            image_files = sorted([item for item in items 
                                if self._is_supported_image_file(item)])
            
            # Add folders first
            for folder_name in folders:
                self._add_folder_item(folder_name, folder_path)
            
            # Add image files
            for filename in image_files:
                self._add_image_item(filename, folder_path)
                
        except Exception as e:
            self.log_with_timestamp(f"❌ Error loading folder: {str(e)}")

    def _is_supported_image_file(self, filename):
        """Check if file is a supported image format"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.image_extensions

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

    # 🚀 NEW: Enhanced file detection with production integrity handler
    def on_new_file_detected(self, file_path):
        """Enhanced file detection using production-ready integrity handler"""
        filename = os.path.basename(file_path)
        
        # Check if it's a supported image
        if not self._is_supported_image_file(filename):
            return

        self.log_with_timestamp(f"🆕 New image detected: {filename}")

        # Add to file list if in current folder
        file_dir = os.path.dirname(file_path)
        if file_dir == self.current_path:
            self._add_image_item(filename, file_dir)

        # Register with production file handler (non-blocking)
        if self.file_handler.register_file(file_path):
            self.log_with_timestamp(f"📝 Registered for processing: {filename}")
        else:
            self.log_with_timestamp(f"⚠️ Already processing or failed: {filename}")
        
        # Update status immediately
        self.update_embedding_status()

    # 🚀 NEW: Process file after integrity verification (callback from handler)
    def _process_ready_file(self, file_path: str):
        """Process file after integrity verification"""
        filename = os.path.basename(file_path)
        
        try:
            # Mark as processing in legacy system for compatibility
            self.processing_files[file_path] = True
            
            # Create and start worker
            worker = FaceEmbeddingWorker(file_path, self.allowed_paths)
            worker.signals.finished.connect(self._on_embedding_finished)
            worker.signals.progress.connect(self.update_progress_label)
            
            self.embedding_in_progress += 1
            self.update_embedding_status()
            self.threadpool.start(worker)
            
            self.log_with_timestamp(f"🚀 Started processing: {filename}")
            
        except Exception as e:
            self.log_with_timestamp(f"❌ Error starting processing for {filename}: {e}")
            # Clean up on error
            if file_path in self.processing_files:
                del self.processing_files[file_path]
            self.update_embedding_status()

    # 🚀 NEW: Handle files that failed integrity checks
    def _handle_failed_file(self, file_path: str, reason: str):
        """Handle files that failed integrity checks"""
        filename = os.path.basename(file_path)
        self.log_with_timestamp(f"❌ File processing failed for {filename}: {reason}")
        self.update_embedding_status()

    # 🚀 ENHANCED: Enhanced version of embedding finished handler
    def _on_embedding_finished(self, file_path: str):
        """Enhanced version of your existing embedding finished handler"""
        # Call your existing logic
        self.on_embedding_finished(file_path, [], True)  # Placeholder values
        
        # Clean up from processing files
        if file_path in self.processing_files:
            del self.processing_files[file_path]
        
        self.update_embedding_status()

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

    # 🚀 NEW: Retry failed files functionality
    def retry_failed_files(self):
        """Retry all failed files"""
        status = self.file_handler.get_status()
        failed_count = status['failed']
        
        if failed_count == 0:
            self.log_with_timestamp("✅ No failed files to retry")
            return
        
        # Get list of failed files and retry them
        with self.file_handler._lock:
            failed_files = list(self.file_handler.failed_files)
        
        retried = 0
        for file_path in failed_files:
            if self.file_handler.reset_failed_file(file_path):
                if self.file_handler.register_file(file_path):
                    retried += 1
        
        self.log_with_timestamp(f"🔄 Retrying {retried} failed files")
        self.update_embedding_status()

    # 🚀 NEW: Get comprehensive processing status
    def get_processing_status(self) -> Dict:
        """Get comprehensive processing status"""
        handler_status = self.file_handler.get_status()
        return {
            **handler_status,
            'embedding_in_progress': self.embedding_in_progress,
            'legacy_processing_count': len(self.processing_files)
        }

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
        
        # 🚀 NEW: Clean shutdown of file handler
        self.file_handler.shutdown()
        
        # Wait for running workers to complete
        self.threadpool.waitForDone(3000)  # 3 second timeout
        event.accept()

