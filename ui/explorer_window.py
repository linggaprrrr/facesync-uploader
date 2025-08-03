# ui/explorer_window.py - FULLY OPTIMIZED HIGH PERFORMANCE VERSION

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
    QProgressBar, QGroupBox, QSpinBox, QCheckBox, QListWidget,
    QComboBox, QTabWidget, QSplitter
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
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

# HIGH PERFORMANCE: Import face detection workers
try:
    from utils.face_detector import FaceEmbeddingWorker, OptimizedBatchFaceEmbeddingWorker
    print("✅ Imported workers from utils.face_detection_yunet")
except ImportError:
    try:
        from utils.face_detector import FaceEmbeddingWorker, OptimizedBatchFaceEmbeddingWorker
        print("✅ Imported workers from utils.face_detector")
    except ImportError:
        try:
            from utils.face_detector import FaceEmbeddingWorker, OptimizedBatchFaceEmbeddingWorker
            print("✅ Imported workers from face_detection_yunet")
        except ImportError:
            print("❌ Warning: Face detection workers not found")
            FaceEmbeddingWorker = None
            OptimizedBatchFaceEmbeddingWorker = None

logger = logging.getLogger(__name__)

class HighPerformanceFileChecker(QThread):
    """Ultra-fast file checker with multiple validation modes"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    
    def __init__(self, file_path, validation_mode="fast", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.validation_mode = validation_mode
        
    def run(self):
        try:
            if self.validation_mode == "instant":
                self._instant_check()
            elif self.validation_mode == "fast":
                self._fast_check()
            elif self.validation_mode == "balanced":
                self._balanced_check()
            else:  # thorough
                self._thorough_check()
                
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Validation error: {e}")

    def _instant_check(self):
        """Instant validation - no waiting"""
        try:
            if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
                self.file_ready.emit(self.file_path)
            else:
                self.file_failed.emit(self.file_path, "Instant check failed")
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Instant check error: {e}")

    def _fast_check(self):
        """Fast validation - 1 second wait"""
        try:
            self.msleep(1000)  # 1 second only
            
            if not os.path.exists(self.file_path):
                self.file_failed.emit(self.file_path, "File not found")
                return
                
            size = os.path.getsize(self.file_path)
            if size == 0:
                self.msleep(1000)  # Wait 1 more second for empty files
                size = os.path.getsize(self.file_path)
                
            if size > 0:
                self.file_ready.emit(self.file_path)
            else:
                self.file_failed.emit(self.file_path, "Fast validation failed")
                
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Fast check error: {e}")

    def _balanced_check(self):
        """Balanced validation - 3 second wait"""
        try:
            self.msleep(2000)  # 2 seconds wait
            
            if not os.path.exists(self.file_path):
                self.file_failed.emit(self.file_path, "File not found")
                return
                
            initial_size = os.path.getsize(self.file_path)
            if initial_size == 0:
                self.msleep(2000)
                initial_size = os.path.getsize(self.file_path)
                
            if initial_size > 0:
                # Check stability
                self.msleep(1000)
                current_size = os.path.getsize(self.file_path)
                
                if current_size == initial_size:
                    self.file_ready.emit(self.file_path)
                else:
                    self.file_failed.emit(self.file_path, "File still changing")
            else:
                self.file_failed.emit(self.file_path, "Balanced validation failed")
                
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Balanced check error: {e}")

    def _thorough_check(self):
        """Thorough validation - original method"""
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
            
            self.file_failed.emit(self.file_path, "Thorough validation failed")
            
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Thorough check error: {e}")

class TurboFileQueue(QObject):
    """Ultra-high performance file queue with concurrent processing"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    queue_status = pyqtSignal(int, int)  # queue_size, processing_count
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self.processing_count = 0
        self.should_stop = False
        self.max_concurrent = 5        # Process up to 5 files simultaneously
        self.validation_mode = "fast"  # Default validation mode
        
        # Ultra-fast processing timer
        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self._process_multiple)
        self.process_timer.start(25)  # Check every 25ms for ultra-fast response
    
    def set_validation_mode(self, mode):
        """Set validation mode: instant, fast, balanced, thorough"""
        self.validation_mode = mode
        print(f"⚡ Validation mode: {mode.upper()}")
    
    def set_max_concurrent(self, count):
        """Set maximum concurrent file processing"""
        self.max_concurrent = max(1, min(count, 10))  # Between 1-10
        print(f"⚡ Max concurrent processing: {self.max_concurrent}")
    
    def add_file(self, file_path):
        """Add file to processing queue"""
        self.queue.put(file_path)
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        return True
    
    def _process_multiple(self):
        """Process multiple files concurrently for maximum performance"""
        if self.should_stop:
            return
        
        # Start new checkers if we have capacity and files
        while (self.processing_count < self.max_concurrent and not self.queue.empty()):
            file_path = self.queue.get()
            self.processing_count += 1
            self.queue_status.emit(self.queue.qsize(), self.processing_count)
            
            # Create high-performance checker
            checker = HighPerformanceFileChecker(file_path, self.validation_mode, self)
            checker.file_ready.connect(self._on_file_ready)
            checker.file_failed.connect(self._on_file_failed)
            checker.start()
    
    def _on_file_ready(self, file_path):
        """Handle file ready"""
        self.processing_count -= 1
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        self.file_ready.emit(file_path)
    
    def _on_file_failed(self, file_path, reason):
        """Handle file failed"""
        self.processing_count -= 1
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        self.file_failed.emit(file_path, reason)
    
    def get_queue_size(self):
        return self.queue.qsize()
    
    def get_processing_count(self):
        return self.processing_count
    
    def stop(self):
        self.should_stop = True
        self.process_timer.stop()

class ExplorerWindow(QMainWindow):
    """ULTRA HIGH PERFORMANCE Explorer Window"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.config_manager = config_manager
        self.setWindowTitle("FaceSync - TURBO MODE 🚀")
        self.setGeometry(100, 100, 1400, 900)
        
        # TURBO PERFORMANCE: File processing
        self.file_queue = TurboFileQueue(self)
        self.file_queue.file_ready.connect(self._on_file_ready)
        self.file_queue.file_failed.connect(self._on_file_failed)
        self.file_queue.queue_status.connect(self._on_queue_status_changed)

        # ULTRA PERFORMANCE MODES
        self.performance_modes = {
            'turbo': {
                'batch_size': 1, 'timeout': 0, 'validation': 'instant',
                'concurrent': 5, 'description': ''
            },
            'speed': {
                'batch_size': 2, 'timeout': 0.5, 'validation': 'fast', 
                'concurrent': 4, 'description': ''
            },
            'stable': {
                'batch_size': 20, 'timeout': 5, 'validation': 'thorough',
                'concurrent': 1, 'description': ''
            }
        }
        
        self.current_mode = 'speed'  # Default to speed mode
        self._apply_performance_mode(self.current_mode)
        
        # Batch processing
        self.use_batch_upload = True
        self.batch_queue = []
        self.batch_processing = False
        
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
        
        self._init_ui()
        self._setup_connections()
        self._load_initial_path()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        # Log startup
        self.log_with_timestamp("🚀 TURBO MODE ACTIVATED")
        self.log_with_timestamp(f"⚡ Mode: {self.current_mode.upper()}")
        self.log_with_timestamp(f"📦 Batch: {self.batch_size}, Timeout: {self.batch_timeout}s")

    def _apply_performance_mode(self, mode):
        """Apply performance mode settings"""
        settings = self.performance_modes[mode]
        self.batch_size = settings['batch_size']
        self.batch_timeout = settings['timeout']
        self.validation_mode = settings['validation']
        self.max_concurrent = settings['concurrent']
        
        # Apply to file queue
        self.file_queue.set_validation_mode(self.validation_mode)
        self.file_queue.set_max_concurrent(self.max_concurrent)

    def _init_ui(self):
        """Initialize ultra-modern UI"""
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
        
        # Search
        # self.search_input = QLineEdit()
        # self.search_input.setPlaceholderText("🔍 Search files...")
        
        # Performance mode selector
        perf_group = QGroupBox("🚀 PERFORMANCE MODES")
        perf_layout = QHBoxLayout()
        
        perf_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(self.performance_modes.keys()))
        self.mode_combo.setCurrentText(self.current_mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        perf_layout.addWidget(self.mode_combo)
        
        self.mode_desc_label = QLabel()
        self.mode_desc_label.setStyleSheet("color: #666; font-style: italic;")
        self._update_mode_description()
        perf_layout.addWidget(self.mode_desc_label)
        
        perf_layout.addStretch()
        perf_group.setLayout(perf_layout)
        
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
        
        self.overall_progress = QLabel("🚀 Ready for turbo processing")
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
        # layout.addWidget(self.search_input)
        layout.addWidget(perf_group)
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
        log_label = QLabel("📋 TURBO LOGS")
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
        # self.search_input.textChanged.connect(self.filter_file_list)

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
        """Handle file ready for processing - TURBO OPTIMIZED"""
        filename = os.path.basename(file_path)
        
        if self.use_batch_upload:
            self.batch_queue.append(file_path)
            self.log_with_timestamp(f"⚡ Added to batch: {filename}")
            
            # TURBO MODE LOGIC
            if self.current_mode == 'turbo':
                # Process immediately
                self.log_with_timestamp(f"🚀 TURBO processing: {filename}")
                self._process_batch()
            elif len(self.batch_queue) >= self.batch_size:
                # Batch full - process immediately
                self.log_with_timestamp(f"📦 Batch full ({self.batch_size}) - processing")
                self._process_batch()
            elif self.batch_timeout > 0:
                # Set timer for timeout
                self._set_batch_timer()
        else:
            # Single file processing
            self.log_with_timestamp(f"🔥 Processing single: {filename}")
            # Could add single file processing here

    def _set_batch_timer(self):
        """Set timer for batch processing"""
        if self.batch_timeout <= 0:
            return  # No timer for instant modes
            
        # Reset timer if already active
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        if not hasattr(self, 'batch_timer'):
            self.batch_timer = QTimer()
            self.batch_timer.timeout.connect(self._auto_process_batch)
            self.batch_timer.setSingleShot(True)
        
        # Start timer
        timeout_ms = int(self.batch_timeout * 1000)
        self.batch_timer.start(timeout_ms)

    def _auto_process_batch(self):
        """AUTO process batch when timer expires"""
        if self.batch_queue and not self.batch_processing:
            self.log_with_timestamp(f"⏰ Auto-processing batch: {len(self.batch_queue)} files")
            self._process_batch()

    def _process_batch(self):
        """Process current batch - TURBO OPTIMIZED"""
        if not self.batch_queue or self.batch_processing:
            return
        
        if not OptimizedBatchFaceEmbeddingWorker:
            self.log_with_timestamp("❌ BatchFaceEmbeddingWorker not available")
            return
        
        # Stop timer
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        batch_files = self.batch_queue.copy()
        self.batch_queue.clear()
        self.batch_processing = True
        
        self.log_with_timestamp(f"🚀 PROCESSING batch: {len(batch_files)} files")
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.overall_progress.setText(f"🚀 Processing {len(batch_files)} files")
        
        # Create and start worker
        worker = OptimizedBatchFaceEmbeddingWorker(batch_files, self.allowed_paths)
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
        """Handle batch completion"""
        self.batch_processing = False
        self.progress_bar.setVisible(False)
        self.overall_progress.setText("🚀 Ready for next batch")
        self.current_operation_label.setText("")
        self.log_with_timestamp(f"✅ Batch completed: {result_summary}")
        
        # AUTO process next batch if files waiting
        if self.batch_queue:
            if self.current_mode == 'turbo':
                self._process_batch()
            else:
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
        """Update performance metrics display"""
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
        self.batch_queue_label.setText(f"Batch: {batch_size}")

    def _on_batch_upload_changed(self, state):
        """Handle batch upload toggle"""
        self.use_batch_upload = state == 2
        mode = "BATCH" if self.use_batch_upload else "SINGLE"
        self.log_with_timestamp(f"⚡ Processing mode: {mode}")
        
        # Process current batch if switching to single mode
        if not self.use_batch_upload and self.batch_queue:
            self.log_with_timestamp("🔄 Switching to single mode - processing current batch")
            self._process_batch()

    def _on_batch_size_changed(self, value):
        """Handle batch size change"""
        self.batch_size = value
        self.log_with_timestamp(f"📦 Batch size: {value}")
        
        # Auto-process if queue >= new size
        if len(self.batch_queue) >= self.batch_size and not self.batch_processing:
            self._process_batch()

    def _on_batch_timeout_changed(self, value):
        """Handle batch timeout change"""
        self.batch_timeout = float(value)
        self.log_with_timestamp(f"⏰ Batch timeout: {value}s")
        
        # Restart timer if active
        if hasattr(self, 'batch_timer') and self.batch_timer.isActive() and self.batch_queue:
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
        """Start folder monitoring - TURBO OPTIMIZED"""
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
                self.log_with_timestamp(f"🔄 Started TURBO monitoring: {os.path.basename(folder)}")
                
                # Log initial status
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
        """Handle new file detection - TURBO MODE"""
        filename = os.path.basename(file_path)
        if self._is_supported_image_file(filename):
            self.log_with_timestamp(f"🆕 New image detected: {filename}")
            if self.file_queue.add_file(file_path):
                self.log_with_timestamp(f"📝 Added to TURBO queue: {filename}")

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
        
        for file_path in failed_list:
            if os.path.exists(file_path):
                self.file_queue.add_file(file_path)
        
        self.log_with_timestamp(f"🔄 TURBO retrying {len(failed_list)} failed files")

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
        self.log_with_timestamp("🔄 Shutting down TURBO mode...")
        self.stop_monitoring()
        self.file_queue.stop()
        if hasattr(self, 'batch_timer'):
            self.batch_timer.stop()
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        self.threadpool.waitForDone(3000)
        self.log_with_timestamp("✅ TURBO mode closed")
        event.accept()