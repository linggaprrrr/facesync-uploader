# ui/explorer_window.py

import sys
import os
import time
import logging
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QListView, QFileDialog, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, 
    QListWidgetItem, QMessageBox, QAbstractItemView, QDialog, 
    QProgressBar, QGroupBox, QSpinBox, QCheckBox, QListWidget,
    QComboBox, QTabWidget, QSplitter
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QThreadPool, QRunnable
import cv2
import numpy as np
from utils.face_detector import OptimizedBatchFaceEmbeddingWorker
from utils.file_queue import TurboFileQueue


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

# HIGH PERFORMANCE: Import face detection workers
try:
    from utils.face_detector import FaceEmbeddingWorker, OptimizedBatchFaceEmbeddingWorker
    print("✅ Imported workers from utils.face_detector")
except ImportError:
    print("❌ Warning: Face detection workers not found")
    FaceEmbeddingWorker = None
    OptimizedBatchFaceEmbeddingWorker = None

logger = logging.getLogger(__name__)

class WorkerSignals(QObject):
    """Worker signals for threading"""
    finished = pyqtSignal(str, bool, str)  # result_summary, success, message
    error = pyqtSignal(str, str)  # file_path, error_message
    progress = pyqtSignal(str, str)  # current_file, status
    batch_completed = pyqtSignal(int, int)  # successful_count, failed_count


class ExplorerWindow(QMainWindow):
    """HIGH PERFORMANCE Explorer Window"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.config_manager = config_manager
        self.setWindowTitle("FaceSync - Uploader")
        self.setGeometry(100, 100, 1400, 900)
        
        # File processing
        self.file_queue = TurboFileQueue(self)
        self.file_queue.file_ready.connect(self._on_file_ready)
        self.file_queue.file_failed.connect(self._on_file_failed)
        self.file_queue.queue_status.connect(self._on_queue_status_changed)

        # PERFORMANCE MODES
        self.performance_modes = {
            'turbo': {
                'batch_size': 10,  # Reduce from 25
                'timeout': 0.5,    # Reduce from 1
                'validation': 'instant',
                'concurrent': 10,  # Increase from 5
                'upload_batch_size': 15,  # Reduce from 20
            },
            'speed': {
                'batch_size': 25, 'timeout': 1, 'validation': 'fast', 
                'concurrent': 4, 'upload_batch_size': 30,
                'description': 'Send every 25 files or 1s'
            },
            'fast': {
                'batch_size': 40, 'timeout': 2, 'validation': 'fast',
                'concurrent': 3, 'upload_batch_size': 50,
                'description': 'Send every 40 files or 2s'
            },
            'balanced': {
                'batch_size': 50, 'timeout': 3, 'validation': 'balanced',
                'concurrent': 2, 'upload_batch_size': 50,
                'description': 'Send every 50 files or 3s'
            },
            'stable': {
                'batch_size': 50, 'timeout': 5, 'validation': 'thorough',
                'concurrent': 1, 'upload_batch_size': 50,
                'description': 'Send every 50 files or 5s'
            }
        }
        
        self.current_mode = 'turbo'  # Default to speed mode
        self._apply_performance_mode(self.current_mode)
        
        # Batch processing
        self.use_batch_upload = True
        self.batch_queue = []
        self.batch_processing = False
        
        # Upload batch size configuration
        self.upload_batch_size = self.performance_modes[self.current_mode]['upload_batch_size']
        
        # State tracking
        self.failed_files = set()
        self.processed_files = set()
        self.watcher_data = None
        
        # Navigation
        self.path_history = []
        self.current_path = ""
        self.allowed_paths = self.config_manager.config.get("allowed_paths", [])
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tga'}
        
        # Performance metrics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self.last_processed_time = time.time()
        
        # Initialize batch timer
        self.batch_timer = None
        
        self._init_ui()
        self._setup_connections()
        self._load_initial_path()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        # Log startup
        self.log_with_timestamp(f"📦 Batch: {self.batch_size}, Timeout: {self.batch_timeout}s")

    def _apply_performance_mode(self, mode):
        """Apply performance mode settings"""
        settings = self.performance_modes[mode]
        self.batch_size = settings['batch_size']
        self.batch_timeout = settings['timeout']
        self.validation_mode = settings['validation']
        self.max_concurrent = settings['concurrent']
        
        # Apply to file queue
        if hasattr(self, 'file_queue'):
            self.file_queue.set_validation_mode(self.validation_mode)
            self.file_queue.set_max_concurrent(self.max_concurrent)

    def _init_ui(self):
        """Initialize modern UI"""
        # Create main splitter
        main_splitter = QSplitter(Qt.Vertical)
        
        # Top section
        top_widget = self._create_top_section()
        main_splitter.addWidget(top_widget)
        
        # Middle section (file list)
        middle_widget = self._create_middle_section()
        main_splitter.addWidget(middle_widget)
        
        # Bottom section (logs)
        bottom_widget = self._create_bottom_section()
        main_splitter.addWidget(bottom_widget)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Top: fixed
        main_splitter.setStretchFactor(1, 1)  # Middle: expandable
        main_splitter.setStretchFactor(2, 0)  # Bottom: fixed
        
        self.setCentralWidget(main_splitter)
        
        # Status bar
        self.status_bar = self.statusBar()
        
        self.embedding_label = QLabel()
        self.status_bar.addPermanentWidget(self.embedding_label)

    def _create_top_section(self):
        """Create top control section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Path and controls
        path_layout = QHBoxLayout()
        self.path_display = QLabel("No path selected")
        self.path_display.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; background: #f0f0f0; border-radius: 3px;")
        
        self.admin_button = QPushButton("⚙️ Admin")
        self.back_button = QPushButton("← Back")
        self.back_button.setEnabled(False)
        self.retry_button = QPushButton("🔄 Retry Failed")
        self.retry_button.clicked.connect(self.retry_failed_files)

        path_layout.addWidget(self.back_button)
        path_layout.addWidget(QLabel("📁"))
        path_layout.addWidget(self.path_display)
        path_layout.addStretch()
        path_layout.addWidget(self.retry_button)
        path_layout.addWidget(self.admin_button)
  
        
        # Batch settings
        batch_group = QGroupBox("📦 BATCH SETTINGS")
        batch_layout = QHBoxLayout()
        
        self.batch_upload_cb = QCheckBox("Batch Upload")
        self.batch_upload_cb.setChecked(True)
        self.batch_upload_cb.stateChanged.connect(self._on_batch_upload_changed)
        batch_layout.addWidget(self.batch_upload_cb)
        
        batch_layout.addWidget(QLabel("Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        batch_layout.addWidget(self.batch_size_spin)
        
        batch_layout.addWidget(QLabel("Timeout:"))
        self.batch_timeout_spin = QSpinBox()
        self.batch_timeout_spin.setRange(0, 60)
        self.batch_timeout_spin.setValue(int(self.batch_timeout))
        self.batch_timeout_spin.setSuffix("s")
        self.batch_timeout_spin.valueChanged.connect(self._on_batch_timeout_changed)
        batch_layout.addWidget(self.batch_timeout_spin)
        
        # Force send button
        
        
        batch_layout.addStretch()
        batch_group.setLayout(batch_layout)
        
        # Performance metrics
        metrics_group = QGroupBox("📊 REAL-TIME METRICS")
        metrics_layout = QVBoxLayout()
        
        metrics_row1 = QHBoxLayout()
        self.processed_label = QLabel("Processed: 0")
        self.failed_label = QLabel("Failed: 0")
        self.rate_label = QLabel("Rate: 0.0/min")
        metrics_row1.addWidget(self.processed_label)
        metrics_row1.addWidget(self.failed_label)
        metrics_row1.addWidget(self.rate_label)
        metrics_row1.addStretch()
        
        metrics_row2 = QHBoxLayout()
        self.queue_label = QLabel("Queue: 0")
        self.processing_label = QLabel("Processing: 0")
        self.batch_queue_label = QLabel("Batch: 0")
        metrics_row2.addWidget(self.queue_label)
        metrics_row2.addWidget(self.processing_label)
        metrics_row2.addWidget(self.batch_queue_label)
        metrics_row2.addStretch()
        
        metrics_layout.addLayout(metrics_row1)
        metrics_layout.addLayout(metrics_row2)
        metrics_group.setLayout(metrics_layout)
        
        # Progress display
        progress_group = QGroupBox("⚡ PROCESSING STATUS")
        progress_layout = QVBoxLayout()
        
        self.overall_progress = QLabel("🚀 App is ready")
        self.overall_progress.setStyleSheet("font-weight: bold; color: #2196F3;")
        progress_layout.addWidget(self.overall_progress)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.current_operation_label = QLabel()
        progress_layout.addWidget(self.current_operation_label)
        
        progress_group.setLayout(progress_layout)
        
        # Add all to main layout
        layout.addLayout(path_layout)
        # layout.addWidget(perf_group)
        layout.addWidget(batch_group)
        layout.addWidget(metrics_group)
        layout.addWidget(progress_group)
        
        return widget
        
    def _create_middle_section(self):
        """Create file list section"""
        self.file_list = QListWidget(self)
        self.file_list.setViewMode(QListView.IconMode)
        self.file_list.setIconSize(QPixmap(100, 100).size())
        self.file_list.setResizeMode(QListView.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setWordWrap(True)
        self.file_list.setGridSize(QPixmap(140, 140).size())
        return self.file_list

    def _create_bottom_section(self):
        """Create log section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        log_header = QHBoxLayout()
        log_label = QLabel("📋 LOGS")
        log_label.setStyleSheet("font-weight: bold;")
        
        self.clear_logs_btn = QPushButton("🗑️ Clear")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        self.clear_logs_btn.setMaximumWidth(80)
        
        log_header.addWidget(log_label)
        log_header.addStretch()
        log_header.addWidget(self.clear_logs_btn)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
        
        layout.addLayout(log_header)
        layout.addWidget(self.log_text)
        
        return widget

    def _setup_connections(self):
        """Setup signal connections"""
        self.admin_button.clicked.connect(self.show_admin_settings)
        self.back_button.clicked.connect(self.go_back)
        self.file_list.itemDoubleClicked.connect(self.open_file)

    def _load_initial_path(self):
        """Load initial path if available"""
        if self.allowed_paths:
            initial_path = self.allowed_paths[0]
            self.set_current_path(initial_path)
            self.load_files(initial_path)
            self.start_monitoring(initial_path)

    def _start_performance_monitoring(self):
        """Start performance monitoring"""
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance_metrics)
        self.perf_timer.start(500)  # Update every 500ms

    def _update_mode_description(self):
        """Update mode description"""
        desc = self.performance_modes[self.current_mode]['description']
        self.mode_desc_label.setText(desc)

    def _on_mode_changed(self, mode):
        """Handle performance mode change"""
        self.current_mode = mode
        self._apply_performance_mode(mode)
        self._update_mode_description()
        
        # Update UI controls
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_timeout_spin.setValue(int(self.batch_timeout))
        
        self.log_with_timestamp(f"🚀 SWITCHED TO {mode.upper()} MODE")
        self.log_with_timestamp(f"⚡ Settings: Batch={self.batch_size}, Timeout={self.batch_timeout}s, Validation={self.validation_mode}")
        
        # Process current batch if in turbo mode
        if mode == 'turbo' and self.batch_queue:
            self.log_with_timestamp("🚀 TURBO MODE - Processing current batch immediately")
            self._process_batch()

    def _on_file_ready(self, file_path):
        """Handle file ready - IMMEDIATE SENDING when batch is full"""
        filename = os.path.basename(file_path)
        
        if self.use_batch_upload:
            self.batch_queue.append(file_path)
            current_batch_size = len(self.batch_queue)
            
            self.log_with_timestamp(f"⚡ Added to batch: {filename} (queue: {current_batch_size}/{self.batch_size})")
            
            # IMMEDIATE SENDING LOGIC
            if self.current_mode == 'turbo':
                # TURBO: Send immediately, no batching
                self.log_with_timestamp(f"Sending immediately - {filename}")
                self._process_batch()
                
            elif current_batch_size >= self.batch_size:
                # BATCH FULL: Send immediately when batch size reached
                self.log_with_timestamp(f"📦 BATCH FULL ({self.batch_size} files) - sending immediately!")
                self._process_batch()
                
            elif self.batch_timeout > 0:
                # START/RESTART TIMER: Will send when timeout reached
                self.log_with_timestamp(f"⏱️ Batch timer: will send in {self.batch_timeout}s if no more files")
                self._set_batch_timer()
        else:
            # Single file processing
            self.log_with_timestamp(f"🔥 Processing single: {filename}")
            # TODO: Implement single file processing if needed

    def _set_batch_timer(self):
        """Set/restart timer for batch timeout - IMMEDIATE SENDING on timeout"""
        if self.batch_timeout <= 0:
            return  # No timer for instant modes
            
        # ALWAYS restart timer when new file arrives
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
            self.log_with_timestamp(f"⏱️ Restarting batch timer: {self.batch_timeout}s")
        
        if not self.batch_timer:
            self.batch_timer = QTimer()
            self.batch_timer.timeout.connect(self._auto_process_batch)
            self.batch_timer.setSingleShot(True)
        
        # Convert float seconds to integer milliseconds
        timeout_ms = int(self.batch_timeout * 1000)
        self.batch_timer.start(timeout_ms)

    def _auto_process_batch(self):
        """AUTO send batch when timer expires - IMMEDIATE SENDING"""
        if self.batch_queue and not self.batch_processing:
            batch_size = len(self.batch_queue)
            self.log_with_timestamp(f"⏰ TIMEOUT REACHED - sending {batch_size} files immediately")
            self._process_batch()
        else:
            self.log_with_timestamp("⏰ Timer expired but no files to send")

    def _force_send_batch(self):
        """Force send current batch"""
        if self.batch_queue and not self.batch_processing:
            batch_size = len(self.batch_queue)
            self.log_with_timestamp(f"🚀 FORCE SEND - sending {batch_size} files now")
            self._process_batch()
        else:
            self.log_with_timestamp("📭 No files to send or already processing")

    def _process_batch(self):
        """Process current batch - IMMEDIATE SENDING with optimal upload batch size"""
        if not self.batch_queue or self.batch_processing:
            return
        
        if not OptimizedBatchFaceEmbeddingWorker:
            self.log_with_timestamp("❌ BatchFaceEmbeddingWorker not available")
            return
        
        # Stop timer since we're processing now
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        # Take current batch for processing
        batch_files = self.batch_queue.copy()
        self.batch_queue.clear()  # Clear queue immediately for new files
        self.batch_processing = True
        
        self.log_with_timestamp(f"🚀 SENDING BATCH: {len(batch_files)} files (upload chunks: {self.upload_batch_size})")
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.overall_progress.setText(f"🚀 Sending {len(batch_files)} files")
        
        # Create and start worker with optimal upload batch size
        worker = OptimizedBatchFaceEmbeddingWorker(
            batch_files, 
            self.allowed_paths, 
            max_upload_batch_size=self.upload_batch_size
        )
        worker.signals.progress.connect(self._on_batch_progress)
        worker.signals.finished.connect(self._on_batch_finished)
        worker.signals.error.connect(self._on_batch_error)
        worker.signals.batch_completed.connect(self._on_batch_completed)
        self.threadpool.start(worker)

    def _on_batch_progress(self, current_file, status):
        """Handle batch progress updates"""
        self.current_operation_label.setText(f"🚀 {status}")
        if current_file != "batch":
            filename = os.path.basename(current_file)
            self.log_with_timestamp(f"🔍 Processing: {filename}")

    def _on_batch_finished(self, result_summary, success, message):
        """Handle batch completion - READY FOR NEXT BATCH IMMEDIATELY"""
        self.batch_processing = False
        self.progress_bar.setVisible(False)
        self.overall_progress.setText("🚀 Ready for next batch")
        self.current_operation_label.setText("")
        
        self.log_with_timestamp(f"✅ Batch sent: {result_summary}")
        
        # IMMEDIATE PROCESSING: Check if new files arrived while processing
        if self.batch_queue:
            new_files_count = len(self.batch_queue)
            self.log_with_timestamp(f"📦 Found {new_files_count} new files during upload")
            
            # If new batch is already full, send immediately
            if new_files_count >= self.batch_size:
                self.log_with_timestamp(f"📦 New batch already full ({new_files_count} files) - sending immediately!")
                self._process_batch()
            elif self.batch_timeout > 0:
                # Start timer for partial batch
                self.log_with_timestamp(f"⏱️ Starting timer for {new_files_count} queued files")
                self._set_batch_timer()

    def _on_batch_error(self, file_path, error_message):
        """Handle batch errors"""
        self.batch_processing = False
        self.failed_count += 1
        self.log_with_timestamp(f"❌ Batch error: {error_message}")

    def _on_batch_completed(self, successful_count, failed_count):
        """Handle batch completion statistics"""
        total = successful_count + failed_count
        self.processed_count += successful_count
        self.failed_count += failed_count
        self.last_processed_time = time.time()
        
        self.log_with_timestamp(f"📊 Stats: {successful_count}/{total} successful, {failed_count} failed")

    def _on_queue_status_changed(self, queue_size, processing_count):
        """Handle queue status changes"""
        self.update_embedding_status()

    def _update_performance_metrics(self):
        """Update performance metrics display with immediate sending info"""
        elapsed = time.time() - self.start_time
        rate = (self.processed_count * 60) / elapsed if elapsed > 0 else 0
        
        self.processed_label.setText(f"Processed: {self.processed_count}")
        self.failed_label.setText(f"Failed: {self.failed_count}")
        self.rate_label.setText(f"Rate: {rate:.1f}/min")
        
        queue_size = self.file_queue.get_queue_size()
        processing = self.file_queue.get_processing_count()
        batch_size = len(self.batch_queue)
        
        self.queue_label.setText(f"Queue: {queue_size}")
        self.processing_label.setText(f"Processing: {processing}")
        
        # Show batch queue with immediate sending info
        if batch_size > 0:
            remaining_for_batch = max(0, self.batch_size - batch_size)
            if remaining_for_batch == 0:
                self.batch_queue_label.setText(f"Batch: {batch_size} (READY TO SEND)")
                self.batch_queue_label.setStyleSheet("color: #FF5722; font-weight: bold;")
            else:
                self.batch_queue_label.setText(f"Batch: {batch_size}/{self.batch_size} (need {remaining_for_batch} more)")
                self.batch_queue_label.setStyleSheet("color: #2196F3;")
        else:
            self.batch_queue_label.setText("Batch: Empty")
            self.batch_queue_label.setStyleSheet("color: #666;")

    def _on_batch_upload_changed(self, state):
        """Handle batch upload toggle with immediate sending explanation"""
        self.use_batch_upload = state == 2
        
        if self.use_batch_upload:
            mode_text = f"BATCH MODE (send every {self.batch_size} files)"
            # Process current batch if switching to batch mode
            if self.batch_queue:
                self.log_with_timestamp("🔄 Switching to batch mode - sending current queue")
                self._process_batch()
        else:
            mode_text = "SINGLE MODE (send immediately)"
        
        self.log_with_timestamp(f"⚡ Processing mode: {mode_text}")

    def _on_batch_size_changed(self, value):
        """Handle batch size change - IMMEDIATE EFFECT"""
        old_batch_size = self.batch_size
        self.batch_size = value
        current_queue_size = len(self.batch_queue)
        
        self.log_with_timestamp(f"📦 Batch size changed: {old_batch_size} → {value}")
        
        # If current queue >= new batch size, send immediately
        if current_queue_size >= self.batch_size and not self.batch_processing:
            self.log_with_timestamp(f"📦 Current queue ({current_queue_size}) ≥ new batch size ({value}) - sending immediately!")
            self._process_batch()

    def _on_batch_timeout_changed(self, value):
        """Handle batch timeout change - IMMEDIATE EFFECT"""
        old_timeout = self.batch_timeout
        self.batch_timeout = float(value)
        
        self.log_with_timestamp(f"⏰ Batch timeout changed: {old_timeout}s → {value}s")
        
        # Restart timer with new timeout if we have queued files
        if self.batch_timer and self.batch_timer.isActive() and self.batch_queue:
            self.log_with_timestamp(f"⏱️ Restarting timer with new timeout: {value}s")
            self._set_batch_timer()

    def update_embedding_status(self):
        """Update status display"""
        queue_size = self.file_queue.get_queue_size()
        processing = self.file_queue.get_processing_count()
        batch_size = len(self.batch_queue)
        failed_count = len(self.failed_files)
        
        status_parts = []
        if self.batch_processing:
            status_parts.append("🚀 Processing")
        elif batch_size > 0:
            status_parts.append(f"📦 Batch: {batch_size}")
        if queue_size > 0:
            status_parts.append(f"⏳ Queue: {queue_size}")
        if processing > 0:
            status_parts.append(f"🔄 Active: {processing}")
        if failed_count > 0:
            status_parts.append(f"❌ Failed: {failed_count}")
        
        self.embedding_label.setText(" | ".join(status_parts) if status_parts else "🚀 Ready")

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

    def start_monitoring(self, folder):
        """Start folder monitoring - OPTIMIZED"""
        if not os.path.isdir(folder):
            return
        
        # Stop existing watcher if any
        self.stop_monitoring()
        
        try:
            self.watcher_data = start_watcher(
                folder, 
                self.on_new_file_detected, 
                self.on_file_deleted
            )
            
            if self.watcher_data:
                observer, event_handler = self.watcher_data
                self.log_with_timestamp(f"🔄 Started monitoring: {os.path.basename(folder)}")
                
                # Log initial status if available
                if hasattr(event_handler, 'get_stats'):
                    stats = event_handler.get_stats()
                    self.log_with_timestamp(f"📊 Initial stats - Processed: {stats['processed']}, Pending: {stats['pending']}, Failed: {stats['failed']}")
            else:
                self.log_with_timestamp(f"❌ Failed to start monitoring: {folder}")
                
        except Exception as e:
            self.log_with_timestamp(f"❌ Error starting monitoring: {e}")

    def stop_monitoring(self):
        """Stop monitoring"""
        if hasattr(self, 'watcher_data') and self.watcher_data:
            try:
                stop_watcher(self.watcher_data)
                self.watcher_data = None
                self.log_with_timestamp("🔄 Stopped folder monitoring")
            except Exception as e:
                self.log_with_timestamp(f"❌ Error stopping monitoring: {e}")

    def on_new_file_detected(self, file_path):
        """Handle new file detection - MODE"""
        filename = os.path.basename(file_path)
        if self._is_supported_image_file(filename):
            self.log_with_timestamp(f"🆕 New image detected: {filename}")
            if self.file_queue.add_file(file_path):
                self.log_with_timestamp(f"📝 Added to queue: {filename}")

    def on_file_deleted(self, file_path):
        """Handle file deletion"""
        filename = os.path.basename(file_path)
        self.processed_files.discard(file_path)
        self.failed_files.discard(file_path)
        self.log_with_timestamp(f"🗑️ File deleted: {filename}")

    def _on_file_failed(self, file_path, reason):
        """Handle file processing failure"""
        filename = os.path.basename(file_path)
        self.failed_files.add(file_path)
        self.failed_count += 1
        self.log_with_timestamp(f"❌ File failed: {filename} - {reason}")

    def retry_failed_files(self):
        """Retry failed files"""
        if not self.failed_files:
            self.log_with_timestamp("✅ No failed files to retry")
            return
        
        failed_list = list(self.failed_files)
        self.failed_files.clear()
        
        retry_count = 0
        for file_path in failed_list:
            if os.path.exists(file_path):
                if self.file_queue.add_file(file_path):
                    retry_count += 1
        
        self.log_with_timestamp(f"🔄 Retrying {retry_count} failed files")

    def open_file(self, item):
        """Open file"""
        item_name = self.get_actual_filename(item)
        if not item_name:
            return
            
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
        if self.path_history:
            previous_path = self.path_history.pop()
            self.set_current_path(previous_path)
            self.load_files(previous_path)
            self.start_monitoring(previous_path)
            self.back_button.setEnabled(len(self.path_history) > 0)

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

    def _clear_logs(self):
        """Clear log display"""
        self.log_text.clear()
        self.log_with_timestamp("🗑️ Logs cleared")

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

    def closeEvent(self, event):
        """Handle close event"""
        self.log_with_timestamp("🔄 Shutting down...")
        self.stop_monitoring()
        self.file_queue.stop()
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        self.threadpool.waitForDone(3000)
        self.log_with_timestamp("✅ Closed")
        event.accept()