# ui/explorer_window.py - Fixed version with correct import

import sys
import os
import time
import logging
from datetime import datetime
from queue import Queue
from PyQt5.QtWidgets import (
    QMainWindow, QListView, QFileDialog, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, 
    QListWidgetItem, QMessageBox, QAbstractItemView, QDialog, 
    QProgressBar, QGroupBox, QSpinBox, QCheckBox, QListWidget
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QThreadPool
import cv2
import numpy as np

# Import dengan try-except untuk avoid errors
try:
    from core.watcher import start_watcher, stop_watcher
except ImportError:
    print("Warning: core.watcher not found")
    def start_watcher(*args, **kwargs): return None
    def stop_watcher(*args, **kwargs): pass

try:
    from ui.admin_login import AdminLoginDialog
    from ui.admin_setting import AdminSettingsDialog
except ImportError:
    print("Warning: Admin dialogs not found")
    AdminLoginDialog = None
    AdminSettingsDialog = None

# FIXED: Updated import path for  workers
try:
    # Try the main face detection module first (where our  code is)
    from utils.face_detection_yunet import FaceEmbeddingWorker, BatchFaceEmbeddingWorker
    print("✅ Imported  workers from utils.face_detection_yunet")
except ImportError:
    try:
        # Fallback to the old path
        from utils.face_detector import FaceEmbeddingWorker, BatchFaceEmbeddingWorker
        print("✅ Imported workers from utils.face_detector")
    except ImportError:
        try:
            # Try importing from the current module if it's in the same directory
            from face_detection_yunet import FaceEmbeddingWorker, BatchFaceEmbeddingWorker
            print("✅ Imported workers from face_detection_yunet")
        except ImportError:
            print("❌ Warning: Face detection workers not found")
            FaceEmbeddingWorker = None
            BatchFaceEmbeddingWorker = None

logger = logging.getLogger(__name__)

class FileQueue(QObject):
    """Simple file queue processor"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    queue_status = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self.processing = False
        self.should_stop = False
        
        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self._process_next)
        self.process_timer.start(100)
    
    def add_file(self, file_path):
        self.queue.put(file_path)
        self.queue_status.emit(self.queue.qsize())
        return True
    
    def _process_next(self):
        if self.processing or self.queue.empty() or self.should_stop:
            return
        
        file_path = self.queue.get()
        self.processing = True
        self.queue_status.emit(self.queue.qsize())
        
        checker = SimpleFileChecker(file_path, self)
        checker.file_ready.connect(self._on_file_ready)
        checker.file_failed.connect(self._on_file_failed)
        checker.start()
    
    def _on_file_ready(self, file_path):
        self.processing = False
        self.file_ready.emit(file_path)
    
    def _on_file_failed(self, file_path, reason):
        self.processing = False
        self.file_failed.emit(file_path, reason)
    
    def get_queue_size(self):
        return self.queue.qsize()
    
    def stop(self):
        self.should_stop = True
        self.process_timer.stop()

class SimpleFileChecker(QThread):
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        
    def run(self):
        try:
            self.msleep(1500)
            
            for attempt in range(5):
                try:
                    if not os.path.exists(self.file_path):
                        self.msleep(1000)
                        continue
                        
                    size = os.path.getsize(self.file_path)
                    if size == 0:
                        self.msleep(1000)
                        continue
                    
                    self.msleep(1000 + (attempt * 500))
                    
                    if os.path.exists(self.file_path) and os.path.getsize(self.file_path) == size:
                        img = cv2.imread(self.file_path)
                        if img is not None and img.size > 0:
                            self.file_ready.emit(self.file_path)
                            return
                    
                except Exception:
                    pass
                
                self.msleep(1000)
            
            self.file_failed.emit(self.file_path, "File validation failed")
            
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Checker error: {e}")

class WatcherThread(QThread):
    new_file_signal = pyqtSignal(str)
    deleted_file_signal = pyqtSignal(str)

    def __init__(self, folder_path, recursive=True):
        super().__init__()
        self.folder_path = folder_path
        self.recursive = recursive
        self.observer = None

    def run(self):
        try:
            self.observer = start_watcher(
                self.folder_path, self.handle_new_file, 
                self.handle_deleted_file, recursive=self.recursive
            )
            self.exec_()
        except Exception as e:
            logger.error(f"Watcher error: {e}")

    def handle_new_file(self, file_path):
        self.new_file_signal.emit(file_path)

    def handle_deleted_file(self, file_path):
        self.deleted_file_signal.emit(file_path)

    def stop(self):
        try:
            if self.observer:
                stop_watcher(self.observer)
            self.quit()
            self.wait()
        except Exception as e:
            logger.error(f"Error stopping watcher: {e}")

class ExplorerWindow(QMainWindow):
    """Minimal Explorer Window dengan  Batch Upload"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.config_manager = config_manager
        self.setWindowTitle("FaceSync -  Batch Uploader")
        self.setGeometry(100, 100, 1200, 800)
        
        # File processing
        self.file_queue = FileQueue(self)
        self.file_queue.file_ready.connect(self._on_file_ready)
        self.file_queue.file_failed.connect(self._on_file_failed)
        self.file_queue.queue_status.connect(self._on_queue_status_changed)

        # Batch settings
        self.use_batch_upload = True
        self.batch_size = 20
        self.batch_timeout = 3
        self.batch_queue = []
        self.batch_processing = False
        
        # State tracking
        self.failed_files = set()
        self.embedding_in_progress = 0
        self.watcher_thread = None
        
        # Navigation
        self.path_history = []
        self.current_path = ""
        self.allowed_paths = self.config_manager.config.get("allowed_paths", [])
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tga'}
        
        self._init_ui()
        self._setup_connections()
        self._load_initial_path()
        
        # Log  status
        if BatchFaceEmbeddingWorker:
            self.log_with_timestamp("✅  BatchFaceEmbeddingWorker loaded successfully")
        else:
            self.log_with_timestamp("❌  BatchFaceEmbeddingWorker not available")

    def _init_ui(self):
        """Initialize UI components"""
        # Top controls
        self.path_display = QLabel("No path selected")
        self.path_display.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        
        self.admin_button = QPushButton("Admin Settings")
        self.back_button = QPushButton("← Back")
        self.back_button.setEnabled(False)
        self.retry_button = QPushButton("🔄 Retry Failed")
        self.retry_button.clicked.connect(self.retry_failed_files)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(QLabel("📁"))
        top_layout.addWidget(self.path_display)
        top_layout.addStretch()
        top_layout.addWidget(self.retry_button)
        top_layout.addWidget(self.admin_button)

        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search files...")

        # Batch controls - Updated for 
        batch_group = QGroupBox("🚀  Auto Batch Upload")
        batch_layout = QHBoxLayout()
        
        self.batch_upload_cb = QCheckBox("Auto Batch Upload")
        self.batch_upload_cb.setChecked(True)
        self.batch_upload_cb.stateChanged.connect(self._on_batch_upload_changed)
        batch_layout.addWidget(self.batch_upload_cb)
        
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(3, 50)
        self.batch_size_spin.setValue(20)
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        batch_layout.addWidget(self.batch_size_spin)
        
        batch_layout.addWidget(QLabel("Auto Timeout:"))
        self.batch_timeout_spin = QSpinBox()
        self.batch_timeout_spin.setRange(3, 60)
        self.batch_timeout_spin.setValue(3)  # Increased default for  processing
        self.batch_timeout_spin.setSuffix("s")
        self.batch_timeout_spin.valueChanged.connect(self._on_batch_timeout_changed)
        batch_layout.addWidget(self.batch_timeout_spin)
        
        batch_layout.addStretch()
        batch_group.setLayout(batch_layout)

        # Progress display - Updated for 
        progress_group = QGroupBox("📊  Upload Progress")
        progress_layout = QVBoxLayout()
        
        self.overall_progress = QLabel("Ready for  uploads")
        progress_layout.addWidget(self.overall_progress)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.current_operation_label = QLabel()
        progress_layout.addWidget(self.current_operation_label)
        
        self.batch_queue_label = QLabel()
        progress_layout.addWidget(self.batch_queue_label)
        
        progress_group.setLayout(progress_layout)

        # File list
        self.file_list = QListWidget(self)
        self.file_list.setViewMode(QListView.IconMode)
        self.file_list.setIconSize(QPixmap(100, 100).size())
        self.file_list.setResizeMode(QListView.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setWordWrap(True)
        self.file_list.setGridSize(QPixmap(140, 140).size())

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.search_input)
        main_layout.addWidget(batch_group)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(self.file_list)
        main_layout.addWidget(QLabel("Logs:"))
        main_layout.addWidget(self.log_text)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(" Ready")
        self.embedding_label = QLabel()
        self.status_bar.addPermanentWidget(self.embedding_label)

    def _setup_connections(self):
        """Setup signal connections"""
        self.admin_button.clicked.connect(self.show_admin_settings)
        self.back_button.clicked.connect(self.go_back)
        self.file_list.itemDoubleClicked.connect(self.open_file)
        self.search_input.textChanged.connect(self.filter_file_list)

    def _load_initial_path(self):
        """Load initial path if available"""
        if self.allowed_paths:
            initial_path = self.allowed_paths[0]
            self.set_current_path(initial_path)
            self.load_files(initial_path)
            self.start_monitoring(initial_path)

    def set_current_path(self, path):
        """Set current path and update display"""
        self.current_path = path
        display_name = os.path.basename(path) if path else ""
        if not display_name:
            display_name = path
        self.path_display.setText(display_name)

    def get_actual_filename(self, item):
        """Get actual filename from list item"""
        return item.data(Qt.UserRole) if item else ""

    def _on_queue_status_changed(self, queue_size):
        self.update_embedding_status()

    def _on_file_ready(self, file_path):
        """Handle file ready for processing - FULLY AUTOMATIC with """
        filename = os.path.basename(file_path)
        
        if self.use_batch_upload:
            self.batch_queue.append(file_path)
            self._update_batch_display()
            self.log_with_timestamp(f"📦 Added to  batch: {filename} (queue: {len(self.batch_queue)})")
            
            # AUTO PROCESS - dua kondisi:
            # 1. Batch penuh -> langsung process
            # 2. Batch belum penuh -> set timer untuk auto process
            if len(self.batch_queue) >= self.batch_size:
                self.log_with_timestamp(f"📦  batch full ({self.batch_size} files) - auto processing...")
                self._process_batch()
            else:
                self.log_with_timestamp(f"⏱️  batch timer set - will auto process in {self.batch_timeout}s")
                self._set_batch_timer()
        else:
            # Single file processing
            self.log_with_timestamp(f"🚀 Processing single file with : {filename}")

    def _set_batch_timer(self):
        """Set timer for AUTOMATIC batch processing"""
        # Reset timer jika sudah ada
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        if not hasattr(self, 'batch_timer'):
            self.batch_timer = QTimer()
            self.batch_timer.timeout.connect(self._auto_process_batch)
            self.batch_timer.setSingleShot(True)
        
        # Start timer
        self.batch_timer.start(self.batch_timeout * 1000)
        self.log_with_timestamp(f"⏱️  auto-process timer started: {self.batch_timeout}s")

    def _auto_process_batch(self):
        """AUTO process batch when timer expires"""
        if self.batch_queue and not self.batch_processing:
            self.log_with_timestamp(f"⏰ Auto-processing  batch: {len(self.batch_queue)} files (timeout reached)")
            self._process_batch()
        else:
            self.log_with_timestamp("⏰ Auto-process timer expired but no files to process")

    def _process_batch(self):
        """Process current batch - AUTOMATIC with """
        if not self.batch_queue or self.batch_processing:
            return
        
        # Check if BatchFaceEmbeddingWorker is available
        if not BatchFaceEmbeddingWorker:
            self.log_with_timestamp("❌  BatchFaceEmbeddingWorker not available - cannot process batch")
            return
        
        # Stop timer karena kita sudah process
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        batch_files = self.batch_queue.copy()
        self.batch_queue.clear()
        self.batch_processing = True
        
        self.log_with_timestamp(f"🚀 AUTO processing  batch: {len(batch_files)} files")
        self._update_batch_display()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.overall_progress.setText(f" auto-processing batch: {len(batch_files)} files")
        
        # Create and start  worker
        worker = BatchFaceEmbeddingWorker(batch_files, self.allowed_paths)
        worker.signals.progress.connect(self._on_batch_progress)
        worker.signals.finished.connect(self._on_batch_finished)
        worker.signals.error.connect(self._on_batch_error)
        worker.signals.batch_completed.connect(self._on_batch_completed)
        self.threadpool.start(worker)
        
        self.log_with_timestamp(f"✅  worker started for {len(batch_files)} files")

    def _on_batch_progress(self, current_file, status):
        """Handle batch progress updates"""
        self.current_operation_label.setText(f": {status}")
        if current_file != "batch":
            filename = os.path.basename(current_file)
            self.log_with_timestamp(f"🔍  processing: {filename}")

    def _on_batch_finished(self, result_summary, success, message):
        """Handle batch completion"""
        self.batch_processing = False
        self.progress_bar.setVisible(False)
        self.overall_progress.setText("Ready for next  batch")
        self.current_operation_label.setText("")
        self.log_with_timestamp(f"🏁  AUTO batch completed - {result_summary}: {message}")
        self._update_batch_display()
        
        # AUTO process next batch if ada files baru yang masuk selama processing
        if self.batch_queue:
            self.log_with_timestamp(f"📦 Found {len(self.batch_queue)} new files - setting  auto timer")
            self._set_batch_timer()

    def _on_batch_error(self, file_path, error_message):
        """Handle batch errors"""
        self.batch_processing = False
        self.log_with_timestamp(f"❌  batch error: {error_message}")
        self._update_batch_display()

    def _on_batch_completed(self, successful_count, failed_count):
        """Handle batch completion statistics"""
        total = successful_count + failed_count
        self.log_with_timestamp(f"📊  batch stats: {successful_count}/{total} successful, {failed_count} failed")

    def _update_batch_display(self):
        """Update batch display - AUTOMATIC mode"""
        queue_size = len(self.batch_queue)
        
        if queue_size > 0:
            if self.batch_processing:
                self.batch_queue_label.setText(f"📦 Processing  batch... (new files queued: {queue_size})")
            else:
                remaining_time = ""
                if hasattr(self, 'batch_timer') and self.batch_timer.isActive():
                    remaining_ms = self.batch_timer.remainingTime()
                    remaining_sec = remaining_ms // 1000
                    remaining_time = f" - auto process in {remaining_sec}s"
                
                self.batch_queue_label.setText(f"📦  Batch Queue: {queue_size} files{remaining_time}")
        else:
            if self.batch_processing:
                self.batch_queue_label.setText("📦 Processing  batch...")
            else:
                self.batch_queue_label.setText("📦 Ready for new files ()")

    def _on_batch_upload_changed(self, state):
        """Handle batch upload toggle"""
        self.use_batch_upload = state == 2
        mode = " auto batch upload" if self.use_batch_upload else " single upload"
        self.log_with_timestamp(f"⚙️ Switched to: {mode}")
        
        # Enable/disable batch controls
        self.batch_size_spin.setEnabled(self.use_batch_upload)
        self.batch_timeout_spin.setEnabled(self.use_batch_upload)
        
        # If switching to single upload, process current batch immediately
        if not self.use_batch_upload and self.batch_queue:
            self.log_with_timestamp(f"🔄 Switching to single mode - processing {len(self.batch_queue)} queued files with ")
            self._process_batch()

    def _on_batch_size_changed(self, value):
        """Handle batch size change"""
        self.batch_size = value
        self.log_with_timestamp(f"⚙️  auto batch size: {value}")
        
        # If current queue >= new batch size, auto process
        if len(self.batch_queue) >= self.batch_size and not self.batch_processing:
            self.log_with_timestamp(f"📦 Queue size ({len(self.batch_queue)}) >= new batch size ({value}) -  auto processing")
            self._process_batch()

    def _on_batch_timeout_changed(self, value):
        """Handle batch timeout change"""
        self.batch_timeout = value
        self.log_with_timestamp(f"⚙️  auto batch timeout: {value}s")
        
        # Restart timer dengan timeout baru jika sedang aktif
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive() and self.batch_queue:
            self.log_with_timestamp(f"⏱️ Restarting  timer with new timeout: {value}s")
            self._set_batch_timer()

    def update_embedding_status(self):
        """Update status display"""
        queue_size = self.file_queue.get_queue_size()
        batch_size = len(self.batch_queue)
        failed_count = len(self.failed_files)
        
        status_parts = []
        if self.batch_processing:
            status_parts.append("📦  processing")
        elif batch_size > 0:
            status_parts.append(f"📦  queue: {batch_size}")
        if queue_size > 0:
            status_parts.append(f"⏳ Queue: {queue_size}")
        if failed_count > 0:
            status_parts.append(f"❌ Failed: {failed_count}")
        
        self.embedding_label.setText(" | ".join(status_parts) if status_parts else " Ready")

    def log_with_timestamp(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        QTimer.singleShot(0, lambda: self._safe_log(full_message))

    def _safe_log(self, message):
        """Safely add log message to UI"""
        try:
            self.log_text.append(message)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except:
            pass

    def load_files(self, folder_path):
        """Load files from folder"""
        if not self.config_manager.is_path_allowed(folder_path):
            return
        
        self.file_list.clear()
        if not os.path.exists(folder_path):
            return
            
        try:
            items = os.listdir(folder_path)
            for filename in items:
                if self._is_supported_image_file(filename):
                    self._add_image_item(filename, folder_path)
        except Exception as e:
            self.log_with_timestamp(f"❌ Error loading folder: {str(e)}")

    def _is_supported_image_file(self, filename):
        """Check if file is supported image"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.image_extensions

    def _add_image_item(self, filename, folder_path):
        """Add image item to list"""
        item = QListWidgetItem()
        item.setText(filename)
        item.setData(Qt.UserRole, filename)
        item.setData(Qt.UserRole + 1, "image")
        
        full_path = os.path.join(folder_path, filename)
        try:
            pixmap = QPixmap(full_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item.setIcon(QIcon(scaled_pixmap))
        except:
            pass
        
        self.file_list.addItem(item)

    def open_file(self, item):
        """Open file"""
        item_name = self.get_actual_filename(item)
        item_path = os.path.join(self.current_path, item_name)
        
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

    def go_back(self):
        """Navigate back"""
        pass  # Implement if needed

    def filter_file_list(self, text):
        """Filter file list"""
        pass  # Implement if needed

    def start_monitoring(self, folder):
        """Start folder monitoring"""
        if not os.path.isdir(folder):
            return
        
        self.watcher_thread = WatcherThread(folder)
        self.watcher_thread.new_file_signal.connect(self.on_new_file_detected)
        self.watcher_thread.start()
        self.log_with_timestamp(f"🔄 Started  monitoring: {os.path.basename(folder)}")

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.watcher_thread:
            self.watcher_thread.stop()
            self.watcher_thread = None

    def on_new_file_detected(self, file_path):
        """Handle new file detection"""
        filename = os.path.basename(file_path)
        if self._is_supported_image_file(filename):
            self.log_with_timestamp(f"🆕 New image detected for : {filename}")
            if self.file_queue.add_file(file_path):
                self.log_with_timestamp(f"📝 Added to  queue: {filename}")

    def _on_file_failed(self, file_path, reason):
        """Handle file processing failure"""
        filename = os.path.basename(file_path)
        self.failed_files.add(file_path)
        self.log_with_timestamp(f"❌  file failed: {filename} - {reason}")

    def retry_failed_files(self):
        """Retry failed files"""
        if not self.failed_files:
            self.log_with_timestamp("✅ No failed files to retry")
            return
        
        failed_list = list(self.failed_files)
        self.failed_files.clear()
        
        for file_path in failed_list:
            if os.path.exists(file_path):
                if self.use_batch_upload:
                    self.batch_queue.append(file_path)
                else:
                    self.file_queue.add_file(file_path)
        
        self.log_with_timestamp(f"🔄  AUTO retrying {len(failed_list)} failed files")
        
        # If switching to batch mode and queue is full, auto process
        if self.use_batch_upload and len(self.batch_queue) >= self.batch_size:
            self.log_with_timestamp("📦 Retry caused  batch to be full - auto processing")
            self._process_batch()
        
        self._update_batch_display()

    def show_admin_settings(self):
        """Show admin settings"""
        if AdminLoginDialog:
            login_dialog = AdminLoginDialog(self.config_manager, self)
            if login_dialog.exec_() == QDialog.Accepted:
                if AdminSettingsDialog:
                    settings_dialog = AdminSettingsDialog(self.config_manager, self)
                    settings_dialog.exec_()
        else:
            QMessageBox.information(self, "Info", "Admin dialogs not available.")

    def closeEvent(self, event):
        """Handle close event"""
        self.log_with_timestamp("🔄 Shutting down batch uploader...")
        self.stop_monitoring()
        self.file_queue.stop()
        if hasattr(self, 'batch_timer'):
            self.batch_timer.stop()
        self.threadpool.waitForDone(3000)
        self.log_with_timestamp("✅ batch uploader closed")
        event.accept()