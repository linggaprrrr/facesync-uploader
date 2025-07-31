import sys
import os
import time
import threading
import logging
from datetime import datetime
from queue import Queue
from PyQt5.QtWidgets import (
    QMainWindow, QListView,
    QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QListWidgetItem, QMessageBox, QAbstractItemView, QDialog, QProgressBar
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from core.watcher import start_watcher, stop_watcher
from ui.admin_login import AdminLoginDialog
from ui.admin_setting import AdminSettingsDialog
from ui.features import DragDropListWidget
from PyQt5.QtCore import QThreadPool
import cv2
import numpy as np
from utils.face_detector import FaceEmbeddingWorker

logger = logging.getLogger(__name__)

class FileQueue(QObject):
    """Simple file queue processor - processes one file at a time"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    queue_status = pyqtSignal(int)  # queue size
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self.processing = False
        self.should_stop = False
        
        # Start processor timer
        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self._process_next)
        self.process_timer.start(100)  # Check every 100ms
    
    def add_file(self, file_path):
        """Add file to processing queue"""
        if not self.queue.empty():
            # Check if file already in queue
            temp_list = []
            while not self.queue.empty():
                item = self.queue.get()
                temp_list.append(item)
                if item == file_path:
                    # File already in queue, put everything back
                    for temp_item in temp_list:
                        self.queue.put(temp_item)
                    return False
            
            # Put everything back
            for temp_item in temp_list:
                self.queue.put(temp_item)
        
        self.queue.put(file_path)
        self.queue_status.emit(self.queue.qsize())
        return True
    
    def _process_next(self):
        """Process next file in queue"""
        if self.processing or self.queue.empty() or self.should_stop:
            return
        
        file_path = self.queue.get()
        self.processing = True
        self.queue_status.emit(self.queue.qsize())
        
        # Start file check in thread
        checker = SimpleFileChecker(file_path, self)
        checker.file_ready.connect(self._on_file_ready)
        checker.file_failed.connect(self._on_file_failed)
        checker.start()
    
    def _on_file_ready(self, file_path):
        """Handle file ready"""
        self.processing = False
        self.file_ready.emit(file_path)
    
    def _on_file_failed(self, file_path, reason):
        """Handle file failed"""
        self.processing = False
        self.file_failed.emit(file_path, reason)
    
    def get_queue_size(self):
        """Get current queue size"""
        return self.queue.qsize()
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
        self.process_timer.stop()

class SimpleFileChecker(QThread):
    """Simple file checker that works reliably"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        
    def run(self):
        """Check file with simple, reliable approach"""
        try:
            # Initial delay to let file finish writing
            self.msleep(800)  # 800ms delay
            
            # Quick validation attempts
            for attempt in range(3):  # Reduced to 3 attempts
                try:
                    # Basic checks
                    if not os.path.exists(self.file_path):
                        self.msleep(500)
                        continue
                        
                    size = os.path.getsize(self.file_path)
                    if size == 0:
                        self.msleep(500)
                        continue
                    
                    # Size stability check
                    self.msleep(200)
                    if os.path.getsize(self.file_path) != size:
                        self.msleep(300)
                        continue
                    
                    # Try reading with OpenCV
                    img = cv2.imread(self.file_path)
                    if img is not None and img.size > 0:
                        # Success!
                        self.file_ready.emit(self.file_path)
                        return
                    
                except Exception:
                    pass
                
                # Wait before next attempt
                self.msleep(500)
            
            # All attempts failed
            self.file_failed.emit(self.file_path, "File validation failed")
            
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Checker error: {e}")

class WatcherThread(QThread):
    """Enhanced File watcher thread with FTP support"""
    new_file_signal = pyqtSignal(str)
    deleted_file_signal = pyqtSignal(str)

    def __init__(self, folder_path, recursive=True):
        super().__init__()
        self.folder_path = folder_path
        self.recursive = recursive
        self.observer = None
        self.event_handler = None

    def run(self):
        # Use the enhanced watcher
        result = start_watcher(
            self.folder_path,
            self.handle_new_file,
            self.handle_deleted_file,
            recursive=self.recursive
        )
        
        # Handle both old and new return format
        if isinstance(result, tuple):
            self.observer, self.event_handler = result
        else:
            self.observer = result
            self.event_handler = None
            
        self.exec_()

    def handle_new_file(self, file_path):
        self.new_file_signal.emit(file_path)

    def handle_deleted_file(self, file_path):
        self.deleted_file_signal.emit(file_path)

    def stop(self):
        if self.observer:
            if self.event_handler:
                # New enhanced format
                stop_watcher((self.observer, self.event_handler))
            else:
                # Old format (backward compatibility)
                stop_watcher(self.observer)
        self.quit()
        self.wait()

class ExplorerWindow(QMainWindow):
    """Queue-based Explorer Window - Processes files one at a time"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)  # Process only 1 file at a time
        self.config_manager = config_manager
        self.setWindowTitle("FaceSync - Uploader")
        self.setGeometry(100, 100, 1200, 700)
        self.watcher_thread = None

        # Initialize file queue processor
        self.file_queue = FileQueue(self)
        self.file_queue.file_ready.connect(self._on_file_ready)
        self.file_queue.file_failed.connect(self._on_file_failed)
        self.file_queue.queue_status.connect(self._on_queue_status_changed)

        # Simple state tracking (only failed_files for queue system)
        self.failed_files = set()
        self.current_processing_file = None
        
        # Initialize UI
        self._init_ui()
        self._setup_connections()
        
        # Status tracking
        self.embedding_in_progress = 0
        
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

        # Retry failed files button
        self.retry_button = QPushButton("🔄 Retry Failed")
        self.retry_button.clicked.connect(self.retry_failed_files)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(QLabel("📁"))
        top_layout.addWidget(self.path_display)
        top_layout.addStretch()
        top_layout.addWidget(self.retry_button)
        top_layout.addWidget(self.admin_button)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search files...")

        # File list
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

        # Queue status label
        self.queue_label = QLabel()

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
        main_layout.addWidget(self.queue_label)
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

    def _on_queue_status_changed(self, queue_size):
        """Update queue status display"""
        if queue_size > 0:
            self.queue_label.setText(f"📋 Queue: {queue_size} files waiting")
        else:
            self.queue_label.setText("")
        self.update_embedding_status()

    def update_embedding_status(self):
        """Update embedding status display"""
        queue_size = self.file_queue.get_queue_size()
        failed_count = len(self.failed_files)
        
        status_parts = []
        
        if self.embedding_in_progress > 0:
            status_parts.append(f"🧠 Processing: 1 file")
        
        if queue_size > 0:
            status_parts.append(f"📋 Queued: {queue_size}")
            
        if failed_count > 0:
            status_parts.append(f"❌ Failed: {failed_count}")
        
        if status_parts:
            self.embedding_label.setText(" | ".join(status_parts))
            self.progress_bar.setVisible(True)
            if queue_size > 0 or self.embedding_in_progress > 0:
                self.progress_bar.setRange(0, 0)  # Indeterminate
            else:
                self.progress_bar.setVisible(False)
        else:
            self.embedding_label.setText("")
            self.progress_bar.setVisible(False)
            if self.embedding_in_progress == 0:
                self.progress_label.setText("")

    def update_progress_label(self, file_path, status):
        """Update progress label"""
        filename = os.path.basename(file_path)
        self.progress_label.setText(f"{status} {filename}")
        
        # Auto-clear after delay for upload status
        if "📤" in status:  # Upload status
            QTimer.singleShot(2000, lambda: self.progress_label.setText("") if self.embedding_in_progress == 0 else None)

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
        """Thread-safe logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        # Use QTimer to ensure main thread execution
        QTimer.singleShot(0, lambda: self._safe_log(full_message))

    def _safe_log(self, message):
        """Safe logging in main thread"""
        try:
            self.log_text.append(message)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except:
            pass  # Ignore logging errors

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
        """Load files from folder"""
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
        
        # Load thumbnail
        try:
            pixmap = QPixmap(full_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item.setIcon(QIcon(scaled_pixmap))
            else:
                item.setIcon(self.style().standardIcon(self.style().SP_FileIcon))
        except:
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
        """Start folder monitoring with enhanced watcher"""
        if folder is None:
            folder = self.current_path
            
        if not os.path.isdir(folder) or not self.config_manager.is_path_allowed(folder):
            return
        
        if (self.watcher_thread and 
            hasattr(self.watcher_thread, 'folder_path') and 
            self.watcher_thread.folder_path == folder):
            return
        
        # Stop existing watcher
        if self.watcher_thread:
            self.stop_monitoring()
        
        self.watcher_thread = WatcherThread(folder, recursive=True)
        self.watcher_thread.new_file_signal.connect(self.on_new_file_detected)
        self.watcher_thread.deleted_file_signal.connect(self.on_file_deleted)
        self.watcher_thread.start()
        
        self.status_bar.showMessage(f"🔄 Monitoring: {os.path.basename(folder)} (FTP compatible)")
        self.log_with_timestamp(f"🔄 Started monitoring with FTP support: {os.path.basename(folder)}")

    def stop_monitoring(self):
        """Stop monitoring with proper cleanup"""
        if self.watcher_thread:
            self.watcher_thread.stop()
            self.watcher_thread = None
            self.status_bar.showMessage("Ready")
            self.log_with_timestamp("⏹️ Stopped monitoring")


    def on_new_file_detected(self, file_path):
        """Enhanced file detection handler"""
        filename = os.path.basename(file_path)
        
        # Check if it's a supported image
        if not self._is_supported_image_file(filename):
            return

        # Determine detection method based on logs
        detection_method = "📡 FTP" if "periodic scan" in str(file_path) else "👁️ Live"
        self.log_with_timestamp(f"🆕 New image detected ({detection_method}): {filename}")

        # Add to file list if in current folder
        file_dir = os.path.dirname(file_path)
        if file_dir == self.current_path:
            self._add_image_item(filename, file_dir)

        # Add to queue (will be processed one at a time)
        if self.file_queue.add_file(file_path):
            self.log_with_timestamp(f"📝 Added to queue: {filename}")
        else:
            self.log_with_timestamp(f"⚠️ Already in queue: {filename}")

    def _on_file_ready(self, file_path):
        """Handle file ready for processing (from queue)"""
        filename = os.path.basename(file_path)
        self.current_processing_file = file_path
        
        try:
            # Create and start face embedding worker
            worker = FaceEmbeddingWorker(file_path, self.allowed_paths)
            worker.signals.finished.connect(self.on_embedding_finished)
            worker.signals.progress.connect(self.update_progress_label)
            
            self.embedding_in_progress = 1  # Always 1 since we process one at a time
            self.update_embedding_status()
            self.threadpool.start(worker)
            
            self.log_with_timestamp(f"🚀 Started processing: {filename}")
            
        except Exception as e:
            self.log_with_timestamp(f"❌ Error starting processing for {filename}: {e}")
            self.failed_files.add(file_path)
            self.current_processing_file = None
            self.update_embedding_status()

    def _on_file_failed(self, file_path, reason):
        """Handle file check failure (from queue)"""
        filename = os.path.basename(file_path)
        self.failed_files.add(file_path)
        self.log_with_timestamp(f"❌ File check failed for {filename}: {reason}")
        self.update_embedding_status()

    def on_embedding_finished(self, file_path, embeddings, success):
        """Handle embedding completion"""
        self.embedding_in_progress = 0
        self.current_processing_file = None
        self.update_embedding_status()
        
        # Clear progress label when nothing is being processed
        if self.embedding_in_progress == 0:
            self.progress_label.setText("")
        
        filename = os.path.basename(file_path)
        if success and embeddings:
            self.log_with_timestamp(f"✅ Processed: {filename} ({len(embeddings)} faces)")
        else:
            self.log_with_timestamp(f"❌ Failed: {filename}")

    def on_file_deleted(self, file_path):
        """Handle file deletion - cleaned up for queue system"""
        filename = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, self.current_path)
        self.log_with_timestamp(f"🗑️ Deleted: {relative_path}")
        
        # Clean up tracking (only failed_files exists in queue system)
        self.failed_files.discard(file_path)
        
        # Update status
        self.update_embedding_status()
        
        # Remove from current view if applicable
        file_dir = os.path.dirname(file_path)
        if file_dir == self.current_path:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.UserRole) == filename:
                    self.file_list.takeItem(i)
                    break

    def retry_failed_files(self):
        """Retry all failed files by adding them back to queue"""
        if not self.failed_files:
            self.log_with_timestamp("✅ No failed files to retry")
            return
        
        failed_list = list(self.failed_files)
        self.failed_files.clear()
        
        retried = 0
        for file_path in failed_list:
            if os.path.exists(file_path):
                if self.file_queue.add_file(file_path):
                    retried += 1
        
        self.log_with_timestamp(f"🔄 Added {retried} failed files back to queue")
        self.update_embedding_status()

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
        
        # Stop file queue
        self.file_queue.stop()
        
        # Wait for running workers to complete
        self.threadpool.waitForDone(3000)  # 3 second timeout
        event.accept()