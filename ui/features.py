
import os
import shutil
from PyQt5.QtWidgets import (
    QListWidget, QMenu, QAction,
    QInputDialog, QMessageBox, QAbstractItemView
)
from PyQt5.QtGui import QDrag
from PyQt5.QtCore import Qt, QMimeData, QUrl






class DragDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.clipboard_files = []  # Store copied files
        
    def contextMenuEvent(self, event):
        selected_items = self.selectedItems()
        menu = QMenu(self)
        
        if selected_items:
            # File-specific actions
            if len(selected_items) == 1:
                copy_action = QAction("Copy", self)
                copy_action.triggered.connect(lambda: self.copy_selected_files())
                menu.addAction(copy_action)
                
                rename_action = QAction("Rename", self)
                rename_action.triggered.connect(lambda: self.rename_file(selected_items[0]))
                menu.addAction(rename_action)
            else:
                copy_action = QAction(f"Copy {len(selected_items)} files", self)
                copy_action.triggered.connect(lambda: self.copy_selected_files())
                menu.addAction(copy_action)
            
            delete_action = QAction(f"Delete {len(selected_items)} file(s)", self)
            delete_action.triggered.connect(lambda: self.delete_selected_files())
            menu.addAction(delete_action)
            
            menu.addSeparator()
        
        # General actions
        if self.clipboard_files:
            paste_action = QAction("Paste", self)
            paste_action.triggered.connect(self.paste_files)
            menu.addAction(paste_action)
        
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_folder)
        menu.addAction(refresh_action)
        
        menu.exec_(event.globalPos())
    
    def get_actual_filename(self, item):
        """Get the actual filename from item data"""
        actual_name = item.data(Qt.UserRole)
        return actual_name if actual_name else item.text()
    
    def get_item_type(self, item):
        """Get the item type (folder or image)"""
        return item.data(Qt.UserRole + 1) or "image"
    
    def copy_selected_files(self):
        if not self.parent_window:
            return
        
        selected_items = self.selectedItems()
        if not selected_items:
            return
            
        current_folder = self.parent_window.current_path
        self.clipboard_files = []
        
        for item in selected_items:
            filename = self.get_actual_filename(item)
            file_path = os.path.join(self.parent_window.current_path, filename)
            if os.path.exists(file_path):
                self.clipboard_files.append(file_path)
        
        if self.clipboard_files:
            self.parent_window.log_text.append(f"📋 Copied {len(self.clipboard_files)} file(s)")
    
    def delete_selected_files(self):
        if not self.parent_window:
            return
        
        selected_items = self.selectedItems()
        if not selected_items:
            return
        
        # Separate folders and files for confirmation message
        folders = []
        files = []
        for item in selected_items:
            if self.get_item_type(item) == "folder":
                folders.append(self.get_actual_filename(item))
            else:
                files.append(self.get_actual_filename(item))
        
        # Create confirmation message
        message_parts = []
        if folders:
            message_parts.append(f"{len(folders)} folder(s)")
        if files:
            message_parts.append(f"{len(files)} file(s)")
        
        confirmation_msg = f"Are you sure you want to delete {' and '.join(message_parts)}?"
        
        reply = QMessageBox.question(self, "Delete Items", confirmation_msg,
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            current_folder = self.parent_window.path_input.text()
            deleted_count = 0
            
            for item in selected_items:
                item_name = self.get_actual_filename(item)
                item_path = os.path.join(current_folder, item_name)
                item_type = self.get_item_type(item)
                
                try:
                    if item_type == "folder":
                        shutil.rmtree(item_path)
                        self.parent_window.log_text.append(f"🗑️ Deleted folder: {item_name}")
                    else:
                        os.remove(item_path)
                        self.parent_window.log_text.append(f"🗑️ Deleted file: {item_name}")
                    deleted_count += 1
                except Exception as e:
                    self.parent_window.log_text.append(f"❌ Error deleting {item_name}: {str(e)}")
            
            if deleted_count > 0:
                self.parent_window.load_files(current_folder)
    
    def copy_file(self, item):
        if self.parent_window:
            current_folder = self.parent_window.path_input.text()
            filename = self.get_actual_filename(item)
            file_path = os.path.join(current_folder, filename)
            if os.path.exists(file_path):
                self.clipboard_files = [file_path]
                self.parent_window.log_text.append(f"📋 Copied: {filename}")
    
    def paste_files(self):
        if not self.clipboard_files or not self.parent_window:
            return
        
        current_folder = self.parent_window.path_input.text()
        for file_path in self.clipboard_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(current_folder, filename)
                
                # Handle duplicate names
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    new_name = f"{base_name}_copy_{counter}{ext}"
                    dest_path = os.path.join(current_folder, new_name)
                    counter += 1
                
                try:
                    if os.path.isdir(file_path):
                        shutil.copytree(file_path, dest_path)
                    else:
                        shutil.copy2(file_path, dest_path)
                    self.parent_window.log_text.append(f"📋 Pasted: {os.path.basename(dest_path)}")
                    self.parent_window.load_files(current_folder)
                except Exception as e:
                    self.parent_window.log_text.append(f"❌ Error pasting {filename}: {str(e)}")
    
    def rename_file(self, item):
        if not self.parent_window:
            return
            
        current_folder = self.parent_window.path_input.text()
        old_filename = self.get_actual_filename(item)
        old_path = os.path.join(current_folder, old_filename)
        
        new_name, ok = QInputDialog.getText(self, "Rename File", "Enter new name:", text=old_filename)
        if ok and new_name and new_name != old_filename:
            new_path = os.path.join(current_folder, new_name)
            try:
                os.rename(old_path, new_path)
                self.parent_window.log_text.append(f"✏️ Renamed: {old_filename} → {new_name}")
                self.parent_window.load_files(current_folder)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not rename file: {str(e)}")
    
    def refresh_folder(self):
        if self.parent_window:
            current_folder = self.parent_window.path_input.text()
            self.parent_window.load_files(current_folder)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_selected_files()
        elif event.key() == Qt.Key_F2:
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                self.rename_file(selected_items[0])
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copy_selected_files()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:
            self.paste_files()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_A:
            self.selectAll()
        else:
            super().keyPressEvent(event)
    
    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()
        if not selected_items or not self.parent_window:
            return
        
        current_folder = self.parent_window.path_input.text()
        valid_paths = []
        
        for item in selected_items:
            item_name = self.get_actual_filename(item)
            item_path = os.path.join(current_folder, item_name)
            if os.path.exists(item_path):
                valid_paths.append(item_path)
        
        if valid_paths:
            drag = QDrag(self)
            mimeData = QMimeData()
            
            # Set file URLs for drag & drop to other applications
            urls = [QUrl.fromLocalFile(path) for path in valid_paths]
            mimeData.setUrls(urls)
            
            # Set text for drag & drop to text applications
            mimeData.setText('\n'.join(valid_paths))
            
            drag.setMimeData(mimeData)
            drag.exec_(Qt.CopyAction)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        if not self.parent_window:
            return
            
        urls = event.mimeData().urls()
        current_folder = self.parent_window.path_input.text()
        
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(current_folder, filename)
                
                # Handle duplicate names
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    new_name = f"{base_name}_copy_{counter}{ext}"
                    dest_path = os.path.join(current_folder, new_name)
                    counter += 1
                
                try:
                    if os.path.isdir(file_path):
                        shutil.copytree(file_path, dest_path)
                    else:
                        shutil.copy2(file_path, dest_path)
                    self.parent_window.log_text.append(f"📁 Dropped: {os.path.basename(dest_path)}")
                except Exception as e:
                    self.parent_window.log_text.append(f"❌ Error dropping {filename}: {str(e)}")
        
        self.parent_window.load_files(current_folder)
        event.acceptProposedAction()