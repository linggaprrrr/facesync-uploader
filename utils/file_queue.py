# utils/file_queue.py - FIXED VERSION

import os
from queue import Queue
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QThreadPool, QRunnable
from utils.file_checker import HighPerformanceFileChecker

class TurboFileQueue(QObject):
    """High performance file queue with PROPER batch processing"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    queue_status = pyqtSignal(int, int)  # queue_size, processing_count
    batch_completed = pyqtSignal(list, list)  # completed_files, failed_files
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self.processing_count = 0
        self.should_stop = False
        self.max_concurrent = 10        # Process up to 10 files simultaneously
        self.batch_size = 20           # Process maksimal 20 file per batch
        self.validation_mode = "instant"  # Default validation mode
        self.active_checkers = []      # Track active checker threads
        
        # Batch tracking
        self.current_batch_count = 0
        self.batch_results = []        # Store completed files in current batch
        self.batch_failed = []         # Store failed files in current batch
        self.is_processing_batch = False
        
        # High-performance processing timer
        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self._process_multiple)
        self.process_timer.start(25)  # Check every 25ms for fast response
    
    def set_validation_mode(self, mode):
        """Set validation mode: instant, fast, balanced, thorough"""
        self.validation_mode = mode
        print(f"⚡ Validation mode: {mode.upper()}")
    
    def set_max_concurrent(self, count):
        """Set maximum concurrent file processing"""
        self.max_concurrent = max(1, min(count, 10))  # Between 1-10
        print(f"⚡ Max concurrent processing: {self.max_concurrent}")
    
    def set_batch_size(self, size):
        """Set batch size (maksimal file yang diproses sebelum dikirim)"""
        self.batch_size = max(1, size)
        print(f"⚡ Batch size: {self.batch_size}")
    
    def add_file(self, file_path):
        """Add file to processing queue"""
        if not self.should_stop:
            self.queue.put(file_path)
            self.queue_status.emit(self.queue.qsize(), self.processing_count)
            return True
        return False
    
    def _process_multiple(self):
        """Process multiple files concurrently with STRICT batch control"""
        if self.should_stop:
            return
        
        # Clean up finished threads safely
        self._cleanup_finished_checkers()
        
        # PERBAIKAN UTAMA: Cek batch limit SEBELUM memproses file baru
        if self.is_processing_batch and self.current_batch_count >= self.batch_size:
            # Sudah mencapai batch limit, tunggu sampai batch selesai
            if len(self.active_checkers) == 0:
                # Semua file dalam batch sudah selesai diproses
                self._complete_batch()
            return  # STOP processing file baru sampai batch selesai
        
        # Start new checkers dengan batch limit control
        while (len(self.active_checkers) < self.max_concurrent and 
               not self.queue.empty() and
               (not self.is_processing_batch or self.current_batch_count < self.batch_size)):
            
            try:
                file_path = self.queue.get_nowait()
                
                # Start new batch if not currently processing one
                if not self.is_processing_batch:
                    self._start_new_batch()
                
                self.processing_count += 1
                self.current_batch_count += 1
                self.queue_status.emit(self.queue.qsize(), self.processing_count)
                
                # Create high-performance checker
                checker = HighPerformanceFileChecker(file_path, self.validation_mode, self)
                checker.file_ready.connect(self._on_file_ready)
                checker.file_failed.connect(self._on_file_failed)
                checker.finished.connect(lambda c=checker: self._on_checker_finished(c))
                
                self.active_checkers.append(checker)
                checker.start()
                
                print(f"📝 Processing batch: {self.current_batch_count}/{self.batch_size}")
                
                # PENTING: Stop kalau sudah mencapai batch limit
                if self.current_batch_count >= self.batch_size:
                    print(f"⛔ BATCH LIMIT REACHED: {self.batch_size} files")
                    break
                
            except:
                break  # Queue is empty
    
    def _start_new_batch(self):
        """Start a new batch"""
        self.is_processing_batch = True
        self.current_batch_count = 0
        self.batch_results.clear()
        self.batch_failed.clear()
        print(f"🚀 Starting new batch (max {self.batch_size} files)")
    
    def _complete_batch(self):
        """Complete current batch and emit results"""
        if not self.is_processing_batch:
            return
            
        self.is_processing_batch = False
        
        print(f"✅ BATCH COMPLETED: {len(self.batch_results)} success, {len(self.batch_failed)} failed")
        
        # Emit batch completion signal
        self.batch_completed.emit(self.batch_results.copy(), self.batch_failed.copy())
        
        # Reset batch tracking
        self.current_batch_count = 0
        self.batch_results.clear()
        self.batch_failed.clear()
        
        # Log remaining queue
        remaining = self.queue.qsize()
        if remaining > 0:
            print(f"📋 Queue remaining: {remaining} files - will start next batch")
        else:
            print(f"🏁 All files processed!")
    
    def _cleanup_finished_checkers(self):
        """Safely clean up finished checker threads"""
        active_checkers = []
        for checker in self.active_checkers:
            if checker.isRunning():
                active_checkers.append(checker)
            else:
                # Thread finished, clean it up
                try:
                    checker.deleteLater()
                except RuntimeError:
                    pass  # Already deleted
        self.active_checkers = active_checkers
    
    def _on_file_ready(self, file_path):
        """Handle file ready"""
        self.processing_count = max(0, self.processing_count - 1)
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        
        # Add to batch results
        self.batch_results.append(file_path)
        
        # Emit individual file ready (optional, untuk compatibility)
        self.file_ready.emit(file_path)
        
        filename = os.path.basename(file_path)
        print(f"✅ File ready: {filename} ({len(self.batch_results)}/{self.batch_size})")
        
        # Check if batch is complete
        self._check_batch_completion()
    
    def _on_file_failed(self, file_path, reason):
        """Handle file failed"""
        self.processing_count = max(0, self.processing_count - 1)
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        
        # Add to batch failed results
        self.batch_failed.append((file_path, reason))
        
        # Emit individual file failed (optional, untuk compatibility)
        self.file_failed.emit(file_path, reason)
        
        filename = os.path.basename(file_path)
        print(f"❌ File failed: {filename} - {reason}")
        
        # Check if batch is complete
        self._check_batch_completion()
    
    def _check_batch_completion(self):
        """Check if current batch is complete"""
        if not self.is_processing_batch:
            return
            
        total_completed = len(self.batch_results) + len(self.batch_failed)
        
        # Batch complete jika:
        # 1. Sudah mencapai batch_size, ATAU
        # 2. Tidak ada active checkers lagi (semua file selesai diproses)
        if (total_completed >= self.batch_size or 
            (len(self.active_checkers) == 0 and total_completed > 0)):
            
            print(f"🎯 BATCH READY: {total_completed} files processed")
            self._complete_batch()
    
    def _on_checker_finished(self, checker):
        """Handle checker thread finished"""
        try:
            if checker in self.active_checkers:
                self.active_checkers.remove(checker)
            
            # Check if batch should be completed
            if self.is_processing_batch and len(self.active_checkers) == 0:
                total_completed = len(self.batch_results) + len(self.batch_failed)
                if total_completed > 0:
                    self._check_batch_completion()
                    
        except (ValueError, RuntimeError):
            pass  # Already removed or deleted
    
    def get_queue_size(self):
        return self.queue.qsize()
    
    def get_processing_count(self):
        return len(self.active_checkers)
    
    def get_batch_status(self):
        """Get current batch status"""
        return {
            'is_processing': self.is_processing_batch,
            'current_count': self.current_batch_count,
            'batch_size': self.batch_size,
            'completed': len(self.batch_results),
            'failed': len(self.batch_failed)
        }
    
    def force_complete_batch(self):
        """Force complete current batch (untuk testing)"""
        if self.is_processing_batch and (self.batch_results or self.batch_failed):
            print(f"🚀 FORCE COMPLETING BATCH: {len(self.batch_results)} files")
            self._complete_batch()
    
    def stop(self):
        """Stop processing and clean up"""
        self.should_stop = True
        self.process_timer.stop()
        
        # Complete current batch if any
        if self.is_processing_batch and (self.batch_results or self.batch_failed):
            self._complete_batch()
        
        # Safely stop and clean up active threads
        for checker in self.active_checkers[:]:  # Copy list to avoid modification during iteration
            try:
                if checker.isRunning():
                    checker.quit()
                    checker.wait(1000)  # Wait up to 1 second
            except RuntimeError:
                pass  # Already deleted
        
        self.active_checkers.clear()

