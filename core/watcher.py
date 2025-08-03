# core/watcher.py - FULLY OPTIMIZED VERSION
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from datetime import datetime

class OptimizedFolderWatcher(FileSystemEventHandler):
    """High-performance folder watcher with duplicate prevention"""
    
    def __init__(self, folder_path, on_new_file, on_delete_file):
        self.folder_path = folder_path
        self.on_new_file = on_new_file
        self.on_delete_file = on_delete_file
        
        # Extended image formats
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp']
        
        # HIGH PERFORMANCE: Separate tracking sets
        self.processed_files = set()    # Successfully processed files
        self.pending_files = set()      # Currently being processed
        self.failed_files = set()       # Failed processing
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance settings
        self.fast_mode = True           # Use fast validation by default
        self.scan_interval = 8          # Scan every 8 seconds (optimized)
        
        # Initialize existing files
        self._scan_existing_files()
        
        # Start optimized periodic scanning
        self.scanning = True
        self.scan_thread = threading.Thread(target=self._optimized_periodic_scan, daemon=True)
        self.scan_thread.start()
        
        print(f"✅ OptimizedFolderWatcher started for: {os.path.basename(folder_path)}")
        print(f"📊 Initial state: {len(self.processed_files)} existing files marked as processed")

    def _scan_existing_files(self):
        """Mark existing files as already processed to prevent reprocessing"""
        try:
            with self.lock:
                for root, dirs, files in os.walk(self.folder_path):
                    for file in files:
                        if self._is_image_file(file):
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                self.processed_files.add(file_path)
                                
        except Exception as e:
            print(f"❌ Error scanning existing files: {e}")

    def _is_image_file(self, filename):
        """Check if file is an image"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.image_extensions

    def _optimized_periodic_scan(self):
        """Optimized periodic scan with better performance"""
        while self.scanning:
            try:
                time.sleep(self.scan_interval)
                self._scan_for_new_files()
                self._cleanup_deleted_files()
            except Exception as e:
                print(f"❌ Error in periodic scan: {e}")
                time.sleep(self.scan_interval)

    def _scan_for_new_files(self):
        """Efficient scan for new files"""
        try:
            new_files_found = 0
            
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if self._is_image_file(file):
                        file_path = os.path.join(root, file)
                        
                        if not os.path.exists(file_path):
                            continue
                            
                        with self.lock:
                            # Skip if already tracked
                            if (file_path in self.processed_files or 
                                file_path in self.pending_files):
                                continue
                        
                        # New file detected
                        new_files_found += 1
                        print(f"🔍 Periodic scan found new file: {os.path.basename(file_path)}")
                        self._handle_new_file(file_path, source="periodic_scan")
            
            if new_files_found > 0:
                print(f"📊 Periodic scan completed: {new_files_found} new files found")
                        
        except Exception as e:
            print(f"❌ Error scanning for new files: {e}")

    def _cleanup_deleted_files(self):
        """Clean up deleted files from tracking sets"""
        try:
            with self.lock:
                all_tracked_files = (self.processed_files | self.pending_files | self.failed_files)
                deleted_files = []
                
                for file_path in all_tracked_files:
                    if not os.path.exists(file_path):
                        deleted_files.append(file_path)
                
                # Remove deleted files from all sets
                for file_path in deleted_files:
                    self.processed_files.discard(file_path)
                    self.pending_files.discard(file_path)
                    self.failed_files.discard(file_path)
                    print(f"🗑️ Cleaned up deleted file: {os.path.basename(file_path)}")
                    # Notify deletion
                    self.on_delete_file(file_path)
                    
        except Exception as e:
            print(f"❌ Error cleaning up deleted files: {e}")

    def _handle_new_file(self, file_path, source="unknown"):
        """Handle new file with duplicate prevention"""
        
        # Prevent duplicate processing
        with self.lock:
            if (file_path in self.processed_files or 
                file_path in self.pending_files):
                print(f"⚠️ DUPLICATE PREVENTED: {os.path.basename(file_path)} (source: {source})")
                return
            
            # Mark as pending
            self.pending_files.add(file_path)
        
        print(f"⚡ Processing new file from {source}: {os.path.basename(file_path)}")
        
        # Process file in background thread
        if self.fast_mode:
            # Fast processing
            processing_thread = threading.Thread(
                target=self._fast_process_file,
                args=(file_path, source),
                daemon=True
            )
        else:
            # Thorough processing
            processing_thread = threading.Thread(
                target=self._thorough_process_file,
                args=(file_path, source),
                daemon=True
            )
        
        processing_thread.start()

    def _fast_process_file(self, file_path, source):
        """Fast file processing - optimized for speed"""
        try:
            # Quick validation only
            if self._fast_file_check(file_path):
                with self.lock:
                    self.pending_files.discard(file_path)
                    self.processed_files.add(file_path)
                
                print(f"✅ FAST: File ready for processing: {os.path.basename(file_path)}")
                self.on_new_file(file_path)
            else:
                with self.lock:
                    self.pending_files.discard(file_path)
                    self.failed_files.add(file_path)
                
                print(f"❌ FAST: File failed validation: {os.path.basename(file_path)}")
                
        except Exception as e:
            with self.lock:
                self.pending_files.discard(file_path)
                self.failed_files.add(file_path)
            print(f"❌ FAST: Error processing {os.path.basename(file_path)}: {e}")

    def _thorough_process_file(self, file_path, source):
        """Thorough file processing - for critical files"""
        try:
            if self._thorough_file_check(file_path):
                with self.lock:
                    self.pending_files.discard(file_path)
                    self.processed_files.add(file_path)
                
                print(f"✅ THOROUGH: File ready for processing: {os.path.basename(file_path)}")
                self.on_new_file(file_path)
            else:
                with self.lock:
                    self.pending_files.discard(file_path)
                    self.failed_files.add(file_path)
                
                print(f"❌ THOROUGH: File failed validation: {os.path.basename(file_path)}")
                
        except Exception as e:
            with self.lock:
                self.pending_files.discard(file_path)
                self.failed_files.add(file_path)
            print(f"❌ THOROUGH: Error processing {os.path.basename(file_path)}: {e}")

    def _fast_file_check(self, file_path):
        """Fast file validation - 1-2 seconds only"""
        try:
            # Wait 1 second
            time.sleep(1)
            
            if not os.path.exists(file_path):
                return False
                
            size = os.path.getsize(file_path)
            if size == 0:
                # Wait a bit more for empty files
                time.sleep(2)
                size = os.path.getsize(file_path)
                
            # File must have content
            return size > 0
            
        except Exception:
            return False

    def _thorough_file_check(self, file_path):
        """Thorough file validation - for critical files"""
        try:
            for attempt in range(3):
                if not os.path.exists(file_path):
                    time.sleep(1)
                    continue
                    
                initial_size = os.path.getsize(file_path)
                if initial_size == 0:
                    time.sleep(2)
                    continue
                
                # Wait for stability
                time.sleep(3 + attempt)
                
                if (os.path.exists(file_path) and 
                    os.path.getsize(file_path) == initial_size and 
                    initial_size > 0):
                    return True
                    
            return False
            
        except Exception:
            return False

    def on_created(self, event):
        """Handle watchdog created event"""
        if not event.is_directory and self._is_image_file(event.src_path):
            print(f"👁️ Watchdog detected: {os.path.basename(event.src_path)}")
            self._handle_new_file(event.src_path, source="watchdog")

    def on_deleted(self, event):
        """Handle watchdog deleted event"""
        if not event.is_directory and self._is_image_file(event.src_path):
            with self.lock:
                self.processed_files.discard(event.src_path)
                self.pending_files.discard(event.src_path)
                self.failed_files.discard(event.src_path)
            print(f"🗑️ Watchdog detected deletion: {os.path.basename(event.src_path)}")
            self.on_delete_file(event.src_path)

    def set_fast_mode(self, enabled):
        """Toggle between fast and thorough processing"""
        self.fast_mode = enabled
        mode = "FAST" if enabled else "THOROUGH"
        print(f"⚡ File processing mode: {mode}")

    def retry_failed_files(self):
        """Retry processing failed files"""
        with self.lock:
            failed_list = list(self.failed_files)
            self.failed_files.clear()
        
        print(f"🔄 Retrying {len(failed_list)} failed files")
        
        for file_path in failed_list:
            if os.path.exists(file_path):
                self._handle_new_file(file_path, source="retry")

    def get_stats(self):
        """Get current processing statistics"""
        with self.lock:
            return {
                'processed': len(self.processed_files),
                'pending': len(self.pending_files),
                'failed': len(self.failed_files),
                'total_tracked': len(self.processed_files) + len(self.pending_files) + len(self.failed_files)
            }

    def stop_scanning(self):
        """Stop the watcher"""
        self.scanning = False
        if self.scan_thread.is_alive():
            self.scan_thread.join(timeout=10)
        print("🔄 OptimizedFolderWatcher stopped")

def start_watcher(folder_path, on_new_file, on_delete_file, recursive=True):
    """
    Start optimized folder watcher
    Returns tuple of (observer, event_handler) for proper cleanup
    """
    try:
        event_handler = OptimizedFolderWatcher(folder_path, on_new_file, on_delete_file)
        observer = Observer()
        observer.schedule(event_handler, folder_path, recursive=recursive)
        observer.start()
        
        print(f"🚀 Started optimized watching: {folder_path}")
        print(f"📡 Watchdog + Periodic scan enabled (recursive={recursive})")
        
        return (observer, event_handler)
        
    except Exception as e:
        print(f"❌ Error starting watcher: {e}")
        return None

def stop_watcher(observer_and_handler):
    """Stop the optimized file watcher"""
    try:
        if isinstance(observer_and_handler, tuple):
            observer, event_handler = observer_and_handler
            # Stop scanning first
            event_handler.stop_scanning()
            # Stop observer
            observer.stop()
            observer.join(timeout=10)
        else:
            # Backward compatibility
            observer = observer_and_handler
            observer.stop()
            observer.join(timeout=10)
            
        print("✅ Optimized file watcher stopped successfully")
        
    except Exception as e:
        print(f"❌ Error stopping watcher: {e}")