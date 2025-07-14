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









class AdminSetupDialog(QDialog):
    """Dialog untuk setup admin pertama kali"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("Setup Admin - File Manager")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Konfigurasi Admin")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Info
        info = QLabel("Aplikasi ini memerlukan setup admin untuk mengatur path yang diizinkan.")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Form
        form_layout = QFormLayout()
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Minimal 6 karakter")
        form_layout.addRow("Password Admin:", self.password_input)
        
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Konfirmasi Password:", self.confirm_password_input)
        
        self.initial_path_input = QLineEdit()
        self.initial_path_input.setPlaceholderText("Contoh: C:\\Users\\Username\\Pictures")
        form_layout.addRow("Path Awal yang Diizinkan:", self.initial_path_input)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_initial_path)
        form_layout.addRow("", browse_btn)
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def browse_initial_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder yang Diizinkan")
        if folder:
            self.initial_path_input.setText(folder)
    
    def accept(self):
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        initial_path = self.initial_path_input.text()
        
        # Validasi
        if len(password) < 6:
            QMessageBox.warning(self, "Error", "Password minimal 6 karakter!")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Password tidak cocok!")
            return
        
        if not initial_path or not os.path.exists(initial_path):
            QMessageBox.warning(self, "Error", "Path tidak valid!")
            return
        
        # Simpan konfigurasi
        if self.config_manager.set_admin_password(password):
            self.config_manager.add_allowed_path(initial_path)
            QMessageBox.information(self, "Sukses", "Konfigurasi admin berhasil disimpan!")
            super().accept()
        else:
            QMessageBox.critical(self, "Error", "Gagal menyimpan konfigurasi!")