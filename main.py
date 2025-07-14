import sys
import os
import shutil
import json
import hashlib
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileSystemModel, QTreeView, QListView,
    QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QAction,
    QInputDialog, QMessageBox, QAbstractItemView, QDialog, QFormLayout,
    QDialogButtonBox, QTabWidget, QGroupBox, QCheckBox
)
from PyQt5.QtGui import QPixmap, QIcon, QDrag, QClipboard
from PyQt5.QtCore import Qt, QDir, QMimeData, QUrl, QThread, pyqtSignal
from watcher import start_watcher, stop_watcher

from admin_setup_dialogs import AdminSetupDialog
from admin_login import AdminLoginDialog
from admin_setting import AdminSettingsDialog
from config_manager import ConfigManager
from explorer_window import ExplorerWindow
from features import DragDropListWidget

class MainApplication:
    """Main application controller"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.config_manager = ConfigManager()
        self.main_window = None
    
    def run(self):
        """Run the application with authentication flow"""
        
        # Check if app is configured
        if not self.config_manager.is_configured():
            # First time setup
            setup_dialog = AdminSetupDialog(self.config_manager)
            if setup_dialog.exec_() != QDialog.Accepted:
                QMessageBox.information(None, "Info", "Setup dibatalkan. Aplikasi akan keluar.")
                return 0
        
        # Check if admin authentication is required
        if self.config_manager.config.get("require_admin", True):
            # Show login dialog
            login_dialog = AdminLoginDialog(self.config_manager)
            if login_dialog.exec_() != QDialog.Accepted:
                QMessageBox.information(None, "Info", "Login diperlukan untuk menggunakan aplikasi.")
                return 0
        
        # Launch main application
        self.main_window = ExplorerWindow(self.config_manager)
        self.main_window.show()
        
        return self.app.exec_()


if __name__ == '__main__':
    app = MainApplication()
    sys.exit(app.run())