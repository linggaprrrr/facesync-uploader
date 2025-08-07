# utils/file_queue.py

import os
from queue import Queue
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QThreadPool, QRunnable
from utils.file_checker import HighPerformanceFileChecker

class TurboFileQueue(QObject):
    """High performance file queue with concurrent processing"""
    
    file_ready = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    queue_status = pyqtSignal(int, int)  # queue_size, processing_count
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self.processing_count = 0
        self.should_stop = False
        self.max_concurrent = 10        # Process up to 10 files simultaneously
        self.validation_mode = "instant"  # Default validation mode
        self.active_checkers = []      # Track active checker threads
        
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
    
    def add_file(self, file_path):
        """Add file to processing queue"""
        if not self.should_stop:
            self.queue.put(file_path)
            self.queue_status.emit(self.queue.qsize(), self.processing_count)
            return True
        return False
    
    def _process_multiple(self):
        """Process multiple files concurrently for maximum performance"""
        if self.should_stop:
            return
        
        # Clean up finished threads safely
        self._cleanup_finished_checkers()
        
        # Start new checkers if we have capacity and files
        while (len(self.active_checkers) < self.max_concurrent and not self.queue.empty()):
            try:
                file_path = self.queue.get_nowait()
                self.processing_count += 1
                self.queue_status.emit(self.queue.qsize(), self.processing_count)
                
                # Create high-performance checker
                checker = HighPerformanceFileChecker(file_path, self.validation_mode, self)
                checker.file_ready.connect(self._on_file_ready)
                checker.file_failed.connect(self._on_file_failed)
                # Use partial to avoid lambda closure issues
                checker.finished.connect(lambda c=checker: self._on_checker_finished(c))
                
                self.active_checkers.append(checker)
                checker.start()
                
            except:
                break  # Queue is empty
    
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
        self.file_ready.emit(file_path)
    
    def _on_file_failed(self, file_path, reason):
        """Handle file failed"""
        self.processing_count = max(0, self.processing_count - 1)
        self.queue_status.emit(self.queue.qsize(), self.processing_count)
        self.file_failed.emit(file_path, reason)
    
    def _on_checker_finished(self, checker):
        """Handle checker thread finished"""
        try:
            if checker in self.active_checkers:
                self.active_checkers.remove(checker)
        except (ValueError, RuntimeError):
            pass  # Already removed or deleted
    
    def get_queue_size(self):
        return self.queue.qsize()
    
    def get_processing_count(self):
        return len(self.active_checkers)
    
    def stop(self):
        """Stop processing and clean up"""
        self.should_stop = True
        self.process_timer.stop()
        
        # Safely stop and clean up active threads
        for checker in self.active_checkers[:]:  # Copy list to avoid modification during iteration
            try:
                if checker.isRunning():
                    checker.quit()
                    checker.wait(1000)  # Wait up to 1 second
                # Don't call deleteLater here - let Qt handle cleanup
            except RuntimeError:
                pass  # Already deleted
        
        self.active_checkers.clear()