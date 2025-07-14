import sys
import os
from datetime import datetime
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
import logging
from PyQt5.QtCore import QThreadPool, pyqtSignal
import logging
from utils.face_detector import FaceEmbeddingWorker
logger = logging.getLogger(__name__)

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
