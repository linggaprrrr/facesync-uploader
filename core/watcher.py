import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from datetime import datetime

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, folder_path, on_new_file, on_delete_file):
        self.folder_path = folder_path
        self.on_new_file = on_new_file
        self.on_delete_file = on_delete_file
        # Extended image formats
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp']
        
        # Track known files untuk periodic scanning
        self.known_files = set()
        self.last_scan_time = datetime.now()
        
        # Initialize known files
        self._scan_existing_files()
        
        # Start periodic scanning thread
        self.scanning = True
        self.scan_thread = threading.Thread(target=self._periodic_scan, daemon=True)
        self.scan_thread.start()

    def _scan_existing_files(self):
        """Scan existing files to populate known_files set"""
        try:
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if self._is_image_file(file):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            self.known_files.add(file_path)
        except Exception as e:
            print(f"Error scanning existing files: {e}")

    def _is_image_file(self, filename):
        """Check if file is an image"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.image_extensions

    def _periodic_scan(self):
        """Periodic scan untuk mendeteksi file yang tidak terdeteksi oleh watchdog"""
        while self.scanning:
            try:
                # Scan setiap 8 detik (lebih lambat untuk mengurangi overhead)
                time.sleep(8)
                self._check_for_new_files()
                self._check_for_deleted_files()
            except Exception as e:
                print(f"Error in periodic scan: {e}")
                time.sleep(8)  # Continue scanning even after error

    def _check_for_new_files(self):
        """Check for new files that weren't detected by watchdog"""
        try:
            current_files = set()
            
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if self._is_image_file(file):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            current_files.add(file_path)
                            
                            # File baru yang belum diketahui
                            if file_path not in self.known_files:
                                # Cek apakah file sudah stabil (selesai transfer)
                                if self._is_file_stable(file_path):
                                    print(f"New file detected via periodic scan: {file_path}")
                                    self.known_files.add(file_path)
                                    self.on_new_file(file_path)
            
            # Update known files list
            # Note: Hanya tambahkan file baru, jangan hapus yang lama di sini
            # karena file deletion akan ditangani oleh _check_for_deleted_files
            
        except Exception as e:
            print(f"Error checking for new files: {e}")

    def _check_for_deleted_files(self):
        """Check for deleted files"""
        try:
            files_to_remove = []
            for file_path in self.known_files:
                if not os.path.exists(file_path):
                    files_to_remove.append(file_path)
                    print(f"File deleted detected via periodic scan: {file_path}")
                    self.on_delete_file(file_path)
            
            # Remove deleted files from known_files
            for file_path in files_to_remove:
                self.known_files.discard(file_path)
                
        except Exception as e:
            print(f"Error checking for deleted files: {e}")

    def _is_file_stable(self, file_path, stable_time=10, max_retries=3):
        """
        Check if file is stable (finished transferring)
        File dianggap stabil jika ukurannya tidak berubah dalam waktu tertentu
        """
        for attempt in range(max_retries):
            try:
                if not os.path.exists(file_path):
                    print(f"File not found on attempt {attempt + 1}: {file_path}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    return False
                    
                # Get initial size and mtime
                initial_size = os.path.getsize(file_path)
                initial_mtime = os.path.getmtime(file_path)
                
                # Skip empty files immediately
                if initial_size == 0:
                    print(f"Empty file detected, waiting: {file_path}")
                    if attempt < max_retries - 1:
                        time.sleep(stable_time)
                        continue
                    return False
                
                # Wait for stable_time seconds
                time.sleep(stable_time)
                
                # Check if file still exists and size/mtime unchanged
                if not os.path.exists(file_path):
                    print(f"File disappeared during stability check: {file_path}")
                    return False
                    
                current_size = os.path.getsize(file_path)
                current_mtime = os.path.getmtime(file_path)
                
                # File is stable if size and mtime unchanged
                if (initial_size == current_size and 
                    initial_mtime == current_mtime and 
                    initial_size > 0):
                    print(f"File stable after {stable_time}s: {os.path.basename(file_path)} ({initial_size} bytes)")
                    return True
                else:
                    print(f"File still changing on attempt {attempt + 1}: {os.path.basename(file_path)} "
                          f"({initial_size} -> {current_size} bytes)")
                    if attempt < max_retries - 1:
                        # Increase wait time for next attempt
                        stable_time += 2
                        continue
                        
            except Exception as e:
                print(f"Error checking file stability for {file_path} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                    
        return False

    def on_created(self, event):
        """Handle watchdog created event"""
        if not event.is_directory:
            if self._is_image_file(event.src_path):
                print(f"New file detected via watchdog: {event.src_path}")
                self.known_files.add(event.src_path)
                self.on_new_file(event.src_path)
    
    def on_deleted(self, event):
        """Handle watchdog deleted event"""
        if not event.is_directory:
            if self._is_image_file(event.src_path):
                print(f"File deleted via watchdog: {event.src_path}")
                self.known_files.discard(event.src_path)
                self.on_delete_file(event.src_path)

    def stop_scanning(self):
        """Stop periodic scanning"""
        self.scanning = False
        if self.scan_thread.is_alive():
            self.scan_thread.join(timeout=10)

def start_watcher(folder_path, on_new_file, on_delete_file, recursive=True):
    """
    Start watching a folder for image file changes
    Includes both watchdog events and periodic scanning for FTP transfers
    
    Args:
        folder_path: Path to watch
        on_new_file: Callback for new files
        on_delete_file: Callback for deleted files  
        recursive: Whether to watch subfolders (default: True)
    """
    event_handler = FolderWatcher(folder_path, on_new_file, on_delete_file)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=recursive)
    observer.start()
    print(f"Started watching: {folder_path} (recursive={recursive})")
    print("Periodic scanning enabled for FTP transfers")
    return observer, event_handler

def stop_watcher(observer_and_handler):
    """Stop the file watcher"""
    if isinstance(observer_and_handler, tuple):
        observer, event_handler = observer_and_handler
        # Stop periodic scanning first
        event_handler.stop_scanning()
        # Stop watchdog observer
        observer.stop()
        observer.join()
    else:
        # Backward compatibility
        observer = observer_and_handler
        observer.stop()
        observer.join()
    print("File watcher stopped")