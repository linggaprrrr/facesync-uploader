# utils/file_checker.py

import os
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QThreadPool, QRunnable

class HighPerformanceFileChecker(QThread):
    """High-performance file checker with multiple validation modes"""
    
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
            self.msleep(500)  
            
            if not os.path.exists(self.file_path):
                self.file_failed.emit(self.file_path, "File not found")
                return
                
            size = os.path.getsize(self.file_path)
            if size == 0:
                self.msleep(500)  
                size = os.path.getsize(self.file_path)
                
            if size > 0:
                self.file_ready.emit(self.file_path)
            else:
                self.file_failed.emit(self.file_path, "Fast validation failed")
                
        except Exception as e:
            self.file_failed.emit(self.file_path, f"Fast check error: {e}")
            