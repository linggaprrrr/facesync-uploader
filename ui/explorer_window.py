# ui/explorer_window.py

import sys
import os
import time
import logging
from datetime import datetime
import asyncio
from typing import List, Dict, Any

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
from utils.file_queue import TurboFileQueue
from core.device_setup import API_BASE

# Import separated upload system
try:
    from utils.face_detector import process_faces_in_image_optimized
    from utils.separated_uploader import SeparatedUploadManager, UploadResult
    print("✅ Imported upload system")
except ImportError:
    print("❌ Warning: upload system not found, using fallback")
    from utils.face_detector import OptimizedBatchFaceEmbeddingWorker
    SeparatedUploadManager = None

# Import with try-except to avoid errors
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

logger = logging.getLogger(__name__)

class SeparatedUploadWorkerSignals(QObject):
    """Signals for separated upload worker"""
    finished = pyqtSignal(str, bool, str)  # result_summary, success, message
    progress = pyqtSignal(str, str)  # current_operation, status
    error = pyqtSignal(str, str)  # file_path, error_message
    batch_completed = pyqtSignal(int, int)  # successful_count, failed_count
    upload_progress = pyqtSignal(int, int)  # current, total


class SeparatedUploadWorker(QRunnable):
    """Worker that uses the new separated upload system"""
    
    def __init__(self, files_list: List[str], allowed_paths: List[str], api_base_url: str = API_BASE):
        super().__init__()
        self.files_list = files_list
        self.allowed_paths = allowed_paths
        self.api_base_url = api_base_url
        self.signals = SeparatedUploadWorkerSignals()
        
    def run(self):
        """Run the separated upload process"""
        try:
            thread_name = f"Worker-{len(self.files_list)}"
            logger.info(f"🚀 [{thread_name}] Starting upload: {len(self.files_list)} files")
            
            # Progress callback
            def progress_callback(message: str, current: int, total: int):
                self.signals.progress.emit(message, f"{current}/{total}")
                self.signals.upload_progress.emit(current, total)
            
            # Process files and extract face data
            self.signals.progress.emit("Processing faces...", "0/0")
            photos_data = []
            processing_errors = 0
            
            for i, file_path in enumerate(self.files_list):
                try:
                    filename = os.path.basename(file_path)
                    self.signals.progress.emit(f"Processing {filename}", f"{i+1}/{len(self.files_list)}")
                    
                    # Process faces in image
                    faces = process_faces_in_image_optimized(file_path)
                    
                    if not faces:
                        processing_errors += 1
                        self.signals.error.emit(file_path, "No faces detected")
                        continue
                    
                    # Parse path codes
                    relative_path = self._get_relative_path(file_path)
                    if not relative_path:
                        processing_errors += 1
                        self.signals.error.emit(file_path, "Invalid path")
                        continue
                    
                    unit_code, outlet_code, photo_type_code = self._parse_codes_from_path(relative_path)
                    if not all([unit_code, outlet_code, photo_type_code]):
                        processing_errors += 1
                        self.signals.error.emit(file_path, "Path parsing failed")
                        continue
                    
                    # Convert codes to IDs (you'll need to implement this based on your system)
                    unit_id, outlet_id, photo_type_id = self._resolve_codes_to_ids(
                        unit_code, outlet_code, photo_type_code
                    )
                    
                    photo_data = {
                        'file_path': file_path,
                        'unit_id': unit_id,
                        'outlet_id': outlet_id,
                        'photo_type_id': photo_type_id,
                        'faces_data': faces
                    }
                    photos_data.append(photo_data)
                    
                except Exception as e:
                    processing_errors += 1
                    self.signals.error.emit(file_path, f"Processing error: {str(e)}")
            
            if not photos_data:
                self.signals.finished.emit("No valid photos to upload", False, "All files failed processing")
                return
            
            # Use separated upload manager
            if SeparatedUploadManager:
                self.signals.progress.emit("Starting upload...", "0/0")
                
                upload_manager = SeparatedUploadManager(self.api_base_url)
                results = upload_manager.upload_photos_sync(photos_data, progress_callback)
                
                # Process results
                successful = len([r for r in results if r.success])
                failed = len([r for r in results if not r.success])
                
                success_rate = (successful / len(results)) * 100 if results else 0
                
                if successful > 0:
                    message = f"Upload: {successful}/{len(results)} successful ({success_rate:.1f}%)"
                    self.signals.finished.emit(message, True, message)
                else:
                    message = f"Upload failed: 0/{len(results)} successful"
                    self.signals.finished.emit(message, False, message)
                
                self.signals.batch_completed.emit(successful, failed)
                
            else:
                # Fallback to old system
                self.signals.progress.emit("Using fallback upload...", "0/0")
                # Implement fallback if needed
                self.signals.finished.emit("Fallback upload not implemented", False, "SeparatedUploadManager not available")
                
        except Exception as e:
            logger.error(f"❌ upload worker error: {e}")
            self.signals.error.emit("batch", f"Worker error: {str(e)}")
            self.signals.finished.emit("Upload failed", False, str(e))
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get relative path from allowed paths"""
        from pathlib import Path
        
        file_path_obj = Path(file_path).resolve()
        
        for root in self.allowed_paths:
            root_path = Path(root).resolve()
            try:
                relative = file_path_obj.relative_to(root_path)
                return str(relative)
            except ValueError:
                continue
        
        return None
    
    def _parse_codes_from_path(self, relative_path: str):
        """Parse codes from relative path"""
        from pathlib import Path
        
        try:
            parts = Path(relative_path).parts
            if len(parts) < 4:
                return None, None, None

            unit_code = parts[0].split("_")[0]
            outlet_code = parts[2].split("_")[0]
            photo_type_code = parts[1].split("_")[0]

            return unit_code, outlet_code, photo_type_code
        except:
            return None, None, None
    
    def _resolve_codes_to_ids(self, unit_code: str, outlet_code: str, photo_type_code: str):
        """
        Convert codes to UUIDs - you'll need to implement this based on your system
        This could be done via API call or local cache
        """
        # TODO: Implement code-to-ID resolution
        # For now, return the codes as IDs (you'll need proper UUID resolution)
        return unit_code, outlet_code, photo_type_code


class ExplorerWindow(QMainWindow):
    """Updated Explorer Window with Upload System"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.threadpool = QThreadPool()
        self.config_manager = config_manager
        self.setWindowTitle("FaceSync - Upload System")
        self.setGeometry(100, 100, 1400, 900)
        
        # API Configuration
        self.api_base_url = API_BASE  # Configure this based on your setup
        
        # File processing
        self.file_queue = TurboFileQueue(self)
        self.file_queue.file_ready.connect(self._on_file_ready)
        self.file_queue.file_failed.connect(self._on_file_failed)
        self.file_queue.queue_status.connect(self._on_queue_status_changed)

        # PERFORMANCE MODES - Updated for separated upload
        self.performance_modes = {
            'turbo': {
                'batch_size': 5,   # Smaller batches for faster data processing
                'timeout': 0.5,
                'validation': 'instant',
                'concurrent': 20,   # More concurrent uploads
                'description': 'Instant processing with separated uploads'
            },
            'speed': {
                'batch_size': 15, 'timeout': 1, 'validation': 'fast', 
                'concurrent': 5,
                'description': 'Fast processing with separated uploads'
            },
            'balanced': {
                'batch_size': 25, 'timeout': 2, 'validation': 'balanced',
                'concurrent': 3,
                'description': 'Balanced processing with separated uploads'
            },
            'stable': {
                'batch_size': 50, 'timeout': 5, 'validation': 'thorough',
                'concurrent': 1,
                'description': 'Stable processing with separated uploads'
            }
        }
        
        self.current_mode = 'turbo'  # Default to turbo mode
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
        
        # Initialize batch timer
        self.batch_timer = None
        
        self._init_ui()
        self._setup_connections()
        self._load_initial_path()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        # Log startup
        self.log_with_timestamp(f"🚀 Upload System Ready")
        self.log_with_timestamp(f"📦 Mode: {self.current_mode.upper()}, Batch: {self.batch_size}, Timeout: {self.batch_timeout}s")
        self.log_with_timestamp(f"🌐 API: {self.api_base_url}")

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
        """Initialize modern UI with separated upload indicators"""
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
        """Create top control section with separated upload info"""
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
        
       
        # Performance metrics
        metrics_group = QGroupBox("📊 STATUS")
        metrics_layout = QVBoxLayout()
        
        metrics_row1 = QHBoxLayout()        
        self.upload_status_label = QLabel("Status: Ready")        
        metrics_row1.addWidget(self.upload_status_label)
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
        
        self.overall_progress = QLabel("🚀 Upload System Ready")
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
        log_label = QLabel("📋 UPLOAD LOGS")
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
        self.mode_desc_label.setStyleSheet("color: #666; font-style: italic;")

    def _on_mode_changed(self, mode):
        """Handle performance mode change"""
        self.current_mode = mode
        self._apply_performance_mode(mode)
        self._update_mode_description()
        
        self.log_with_timestamp(f"🚀 SWITCHED TO {mode.upper()} MODE")
        self.log_with_timestamp(f"⚡ Settings: Batch={self.batch_size}, Timeout={self.batch_timeout}s")
        
        # Process current batch if in turbo mode
        if mode == 'turbo' and self.batch_queue:
            self.log_with_timestamp("🚀 TURBO MODE - Processing current batch immediately")
            self._process_batch()

    def _on_file_ready(self, file_path):
        """Handle file ready - using separated upload system"""
        filename = os.path.basename(file_path)
        
        if self.use_batch_upload:
            self.batch_queue.append(file_path)
            current_batch_size = len(self.batch_queue)
            
            self.log_with_timestamp(f"⚡ Added to batch: {filename} (queue: {current_batch_size}/{self.batch_size})")
            
            # SEPARATED UPLOAD LOGIC
            if self.current_mode == 'turbo':
                # TURBO: Send immediately
                self.log_with_timestamp(f"🚀 TURBO: Sending immediately - {filename}")
                self._process_batch()
                
            elif current_batch_size >= self.batch_size:
                # BATCH FULL: Send immediately when batch size reached
                self.log_with_timestamp(f"📦 BATCH FULL ({self.batch_size} files) - sending with separated upload!")
                self._process_batch()
                
            elif self.batch_timeout > 0:
                # START/RESTART TIMER: Will send when timeout reached
                self.log_with_timestamp(f"⏱️ Batch timer: will send in {self.batch_timeout}s")
                self._set_batch_timer()
        else:
            # Single file processing
            self.log_with_timestamp(f"🔥 Processing single file: {filename}")
            self._process_single_file(file_path)

    def _process_single_file(self, file_path):
        """Process single file with separated upload"""
        files_list = [file_path]
        self._start_separated_upload_worker(files_list)

    def _set_batch_timer(self):
        """Set/restart timer for batch timeout"""
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
        """AUTO send batch when timer expires"""
        if self.batch_queue and not self.batch_processing:
            batch_size = len(self.batch_queue)
            self.log_with_timestamp(f"⏰ TIMEOUT REACHED - sending {batch_size} files with separated upload")
            self._process_batch()
        else:
            self.log_with_timestamp("⏰ Timer expired but no files to send")

    def _force_send_batch(self):
        """Force send current batch"""
        if self.batch_queue and not self.batch_processing:
            batch_size = len(self.batch_queue)
            self.log_with_timestamp(f"🚀 FORCE SEND - sending {batch_size} files with separated upload")
            self._process_batch()
        else:
            self.log_with_timestamp("📭 No files to send or already processing")

    def _process_batch(self):
        """Process current batch using separated upload system"""
        if not self.batch_queue or self.batch_processing:
            return
        
        # Stop timer since we're processing now
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        # Take current batch for processing
        batch_files = self.batch_queue.copy()
        self.batch_queue.clear()  # Clear queue immediately for new files
        self.batch_processing = True
        
        self.log_with_timestamp(f"🚀 SEPARATED UPLOAD: {len(batch_files)} files")
        
        # Start separated upload worker
        self._start_separated_upload_worker(batch_files)

    def _start_separated_upload_worker(self, files_list):
        """Start the separated upload worker"""
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.overall_progress.setText(f"🚀 Upload: {len(files_list)} files")
        self.upload_status_label.setText("Status: Uploading")
        self.upload_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
        # Create and start worker
        worker = SeparatedUploadWorker(files_list, self.allowed_paths, self.api_base_url)
        worker.signals.progress.connect(self._on_separated_upload_progress)
        worker.signals.finished.connect(self._on_separated_upload_finished)
        worker.signals.error.connect(self._on_separated_upload_error)
        worker.signals.batch_completed.connect(self._on_separated_upload_completed)
        worker.signals.upload_progress.connect(self._on_upload_progress)
        self.threadpool.start(worker)

    def _on_separated_upload_progress(self, current_operation, status):
        """Handle separated upload progress updates"""
        self.current_operation_label.setText(f"🚀 {current_operation}")
        if status:
            self.upload_status_label.setText(f"Status: {status}")

    def _on_upload_progress(self, current, total):
        """Handle upload progress bar updates"""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)

    def _on_separated_upload_finished(self, result_summary, success, message):
        """Handle separated upload completion"""
        self.batch_processing = False
        self.progress_bar.setVisible(False)
        self.overall_progress.setText("🚀 Ready for next batch")
        self.current_operation_label.setText("")
        
        if success:
            self.upload_status_label.setText("Status: Success")
            self.upload_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.log_with_timestamp(f"✅ upload completed: {result_summary}")
        else:
            self.upload_status_label.setText("Status: Failed")
            self.upload_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.log_with_timestamp(f"❌ upload failed: {message}")
        
        # Reset status after delay
        QTimer.singleShot(3000, lambda: (
            self.upload_status_label.setText("Status: Ready"),
            self.upload_status_label.setStyleSheet("color: #666;")
        ))
        
        # Check if new files arrived while processing
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

    def _on_separated_upload_error(self, file_path, error_message):
        """Handle separated upload errors"""
        filename = os.path.basename(file_path) if file_path != "batch" else "batch"
        self.failed_count += 1
        # self.log_with_timestamp(f"❌ Upload error ({filename}): {error_message}")

    def _on_separated_upload_completed(self, successful_count, failed_count):
        """Handle separated upload completion statistics"""
        total = successful_count + failed_count
        self.processed_count += successful_count
        self.failed_count += failed_count
        self.last_processed_time = time.time()
        
        self.log_with_timestamp(f"📊 Upload stats: {successful_count}/{total} successful, {failed_count} failed")

    def _on_queue_status_changed(self, queue_size, processing_count):
        """Handle queue status changes"""
        self.update_embedding_status()

    def _update_performance_metrics(self):
        """Update performance metrics display"""
        elapsed = time.time() - self.start_time
        rate = (self.processed_count * 60) / elapsed if elapsed > 0 else 0
        
        
        
        queue_size = self.file_queue.get_queue_size()
        
        batch_size = len(self.batch_queue)
        
        self.queue_label.setText(f"Queue: {queue_size}")
        
        
        # Show batch queue with separated upload info
        if batch_size > 0:
            remaining_for_batch = max(0, self.batch_size - batch_size)
            if remaining_for_batch == 0:
                self.batch_queue_label.setText(f"Batch: {batch_size} (READY FOR SEPARATED UPLOAD)")
                self.batch_queue_label.setStyleSheet("color: #FF5722; font-weight: bold;")
            else:
                self.batch_queue_label.setText(f"Batch: {batch_size}/{self.batch_size} (need {remaining_for_batch} more)")
                self.batch_queue_label.setStyleSheet("color: #2196F3;")
        else:
            self.batch_queue_label.setText("Batch: Empty")
            self.batch_queue_label.setStyleSheet("color: #666;")

    def update_embedding_status(self):
        """Update status display"""
        queue_size = self.file_queue.get_queue_size()
        processing = self.file_queue.get_processing_count()
        batch_size = len(self.batch_queue)
        failed_count = len(self.failed_files)
        
        status_parts = []
        if self.batch_processing:
            status_parts.append("")
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
        """Start folder monitoring"""
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
                self.log_with_timestamp(f"🔄 Started monitoring: {os.path.basename(folder)} (Upload)")
                
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
        """Handle new file detection"""
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
        
        self.log_with_timestamp(f"🔄 Retrying {retry_count} failed files with separated upload")

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
        self.log_with_timestamp("🔄 Shutting down separated upload system...")
        self.stop_monitoring()
        self.file_queue.stop()
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        self.threadpool.waitForDone(3000)
        self.log_with_timestamp("✅ upload system closed")
        event.accept()


# ===== COMPATIBILITY ADAPTER =====

class SeparatedUploadAdapter:
    """
    Adapter to integrate separated upload system with existing codebase
    This allows gradual migration without breaking existing functionality
    """
    
    def __init__(self, api_base_url: str = API_BASE):
        self.api_base_url = api_base_url
        self.upload_manager = None
        
        # Initialize separated upload manager if available
        if SeparatedUploadManager:
            try:
                self.upload_manager = SeparatedUploadManager(api_base_url)
                logger.info("✅ upload adapter initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize separated upload manager: {e}")
                self.upload_manager = None
        else:
            logger.warning("⚠️ SeparatedUploadManager not available, using fallback")
    
    def process_batch_faces_and_upload_separated(self, 
                                               files_list: List[str], 
                                               allowed_paths: List[str],
                                               progress_callback=None) -> tuple:
        """
        Drop-in replacement for existing batch processing functions
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        
        if not self.upload_manager:
            # Fallback to old system if separated upload not available
            logger.warning("⚠️ upload not available, using fallback")
            try:
                from utils.face_detector import process_batch_faces_and_upload_optimized
                return process_batch_faces_and_upload_optimized(files_list, allowed_paths)
            except ImportError:
                return False, "Neither separated upload nor fallback system available"
        
        try:
            logger.info(f"🚀 Processing {len(files_list)} files with separated upload")
            
            # Process files and extract face data
            photos_data = []
            processing_errors = 0
            
            for file_path in files_list:
                try:
                    # Process faces in image
                    faces = process_faces_in_image_optimized(file_path)
                    
                    if not faces:
                        processing_errors += 1
                        continue
                    
                    # Parse path codes (implement based on your system)
                    relative_path = self._get_relative_path(file_path, allowed_paths)
                    if not relative_path:
                        processing_errors += 1
                        continue
                    
                    unit_code, outlet_code, photo_type_code = self._parse_codes_from_path(relative_path)
                    if not all([unit_code, outlet_code, photo_type_code]):
                        processing_errors += 1
                        continue
                    
                    # Convert codes to IDs (you'll need to implement this)
                    unit_id, outlet_id, photo_type_id = self._resolve_codes_to_ids(
                        unit_code, outlet_code, photo_type_code
                    )
                    
                    photo_data = {
                        'file_path': file_path,
                        'unit_id': unit_id,
                        'outlet_id': outlet_id,
                        'photo_type_id': photo_type_id,
                        'faces_data': faces
                    }
                    photos_data.append(photo_data)
                    
                except Exception as e:
                    processing_errors += 1
                    logger.error(f"❌ Processing error for {file_path}: {e}")
            
            if not photos_data:
                return False, f"No valid photos to upload. {processing_errors} processing errors."
            
            # Use separated upload
            results = self.upload_manager.upload_photos_sync(photos_data, progress_callback)
            
            # Process results
            successful = len([r for r in results if r.success])
            failed = len([r for r in results if not r.success])
            
            success_rate = (successful / len(results)) * 100 if results else 0
            
            if successful > 0:
                message = f"upload: {successful}/{len(results)} successful ({success_rate:.1f}%)"
                return True, message
            else:
                message = f"upload failed: 0/{len(results)} successful"
                return False, message
                
        except Exception as e:
            logger.error(f"❌ upload adapter error: {e}")
            return False, f"upload failed: {str(e)}"
    
    def _get_relative_path(self, file_path: str, allowed_paths: List[str]) -> str:
        """Get relative path from allowed paths"""
        from pathlib import Path
        
        file_path_obj = Path(file_path).resolve()
        
        for root in allowed_paths:
            root_path = Path(root).resolve()
            try:
                relative = file_path_obj.relative_to(root_path)
                return str(relative)
            except ValueError:
                continue
        
        return None
    
    def _parse_codes_from_path(self, relative_path: str):
        """Parse codes from relative path"""
        from pathlib import Path
        
        try:
            parts = Path(relative_path).parts
            if len(parts) < 4:
                return None, None, None

            unit_code = parts[0].split("_")[0]
            outlet_code = parts[1].split("_")[0]
            photo_type_code = parts[2].split("_")[0]

            return unit_code, outlet_code, photo_type_code
        except:
            return None, None, None
    
    def _resolve_codes_to_ids(self, unit_code: str, outlet_code: str, photo_type_code: str):
        """
        Convert codes to UUIDs - implement this based on your system
        """
        # TODO: Implement proper code-to-ID resolution
        # This could be done via API call or local cache
        return unit_code, outlet_code, photo_type_code


# ===== GLOBAL ADAPTER INSTANCE =====

# Create global adapter instance for easy integration
separated_upload_adapter = SeparatedUploadAdapter()

def process_batch_faces_and_upload_separated(files_list: List[str], 
                                           allowed_paths: List[str],
                                           progress_callback=None) -> tuple:
    """
    Global function that can be used as drop-in replacement
    for existing batch upload functions
    """
    return separated_upload_adapter.process_batch_faces_and_upload_separated(
        files_list, allowed_paths, progress_callback
    )