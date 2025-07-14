import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, folder_path, on_new_file, on_delete_file):
        self.folder_path = folder_path
        self.on_new_file = on_new_file
        self.on_delete_file = on_delete_file
        # Extended image formats
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp']

    def on_created(self, event):
        if not event.is_directory:
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in self.image_extensions:
                print(f"New file detected: {event.src_path}")
                self.on_new_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in self.image_extensions:
                print(f"File deleted: {event.src_path}")
                self.on_delete_file(event.src_path)

def start_watcher(folder_path, on_new_file, on_delete_file, recursive=True):
    """
    Start watching a folder for image file changes
    
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
    return observer

def stop_watcher(observer):
    """Stop the file watcher"""
    observer.stop()
    observer.join()
    print("File watcher stopped")