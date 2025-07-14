
from PyQt5.QtWidgets import (
    QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QAction,
    QInputDialog, QMessageBox, QAbstractItemView, QDialog, QFormLayout,
    QDialogButtonBox, QTabWidget, QGroupBox, QCheckBox
)

class AdminSettingsDialog(QDialog):
    """Dialog untuk pengaturan admin"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("Pengaturan Admin")
        self.setModal(True)
        self.setFixedSize(500, 400)
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Tab widget
        tabs = QTabWidget()
        
        # Tab 1: Allowed Paths
        paths_tab = QWidget()
        paths_layout = QVBoxLayout()
        
        paths_group = QGroupBox("Path yang Diizinkan")
        paths_group_layout = QVBoxLayout()
        
        self.paths_list = QListWidget()
        paths_group_layout.addWidget(self.paths_list)
        
        paths_buttons_layout = QHBoxLayout()
        add_path_btn = QPushButton("Tambah Path")
        remove_path_btn = QPushButton("Hapus Path")
        add_path_btn.clicked.connect(self.add_path)
        remove_path_btn.clicked.connect(self.remove_path)
        paths_buttons_layout.addWidget(add_path_btn)
        paths_buttons_layout.addWidget(remove_path_btn)
        paths_group_layout.addLayout(paths_buttons_layout)
        
        paths_group.setLayout(paths_group_layout)
        paths_layout.addWidget(paths_group)
        paths_tab.setLayout(paths_layout)
        
        # Tab 2: Security
        security_tab = QWidget()
        security_layout = QVBoxLayout()
        
        security_group = QGroupBox("Keamanan")
        security_group_layout = QVBoxLayout()
        
        change_password_btn = QPushButton("Ubah Password Admin")
        change_password_btn.clicked.connect(self.change_password)
        security_group_layout.addWidget(change_password_btn)
        
        self.require_admin_checkbox = QCheckBox("Memerlukan autentikasi admin")
        self.require_admin_checkbox.stateChanged.connect(self.toggle_admin_requirement)
        security_group_layout.addWidget(self.require_admin_checkbox)
        
        security_group.setLayout(security_group_layout)
        security_layout.addWidget(security_group)
        security_tab.setLayout(security_layout)
        
        tabs.addTab(paths_tab, "Path yang Diizinkan")
        tabs.addTab(security_tab, "Keamanan")
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def load_settings(self):
        # Load allowed paths
        self.paths_list.clear()
        for path in self.config_manager.config.get("allowed_paths", []):
            self.paths_list.addItem(path)
        
        # Load security settings
        self.require_admin_checkbox.setChecked(
            self.config_manager.config.get("require_admin", True)
        )
    
    def add_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder yang Diizinkan")
        if folder:
            if self.config_manager.add_allowed_path(folder):
                self.paths_list.addItem(folder)
                QMessageBox.information(self, "Sukses", f"Path berhasil ditambahkan: {folder}")
            else:
                QMessageBox.warning(self, "Error", "Gagal menambahkan path!")
    
    def remove_path(self):
        current_item = self.paths_list.currentItem()
        if current_item:
            path = current_item.text()
            reply = QMessageBox.question(self, "Konfirmasi", 
                                       f"Hapus path ini?\n{path}",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.config_manager.remove_allowed_path(path):
                    self.paths_list.takeItem(self.paths_list.row(current_item))
                    QMessageBox.information(self, "Sukses", "Path berhasil dihapus!")
                else:
                    QMessageBox.warning(self, "Error", "Gagal menghapus path!")
    
    def change_password(self):
        # Dialog untuk ubah password
        dialog = QDialog(self)
        dialog.setWindowTitle("Ubah Password")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        old_password = QLineEdit()
        old_password.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password Lama:", old_password)
        
        new_password = QLineEdit()
        new_password.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password Baru:", new_password)
        
        confirm_password = QLineEdit()
        confirm_password.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Konfirmasi Password:", confirm_password)
        
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        
        def change_pwd():
            if not self.config_manager.verify_password(old_password.text()):
                QMessageBox.warning(dialog, "Error", "Password lama salah!")
                return
            
            if len(new_password.text()) < 6:
                QMessageBox.warning(dialog, "Error", "Password baru minimal 6 karakter!")
                return
            
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "Password baru tidak cocok!")
                return
            
            if self.config_manager.set_admin_password(new_password.text()):
                QMessageBox.information(dialog, "Sukses", "Password berhasil diubah!")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Error", "Gagal mengubah password!")
        
        buttons.accepted.connect(change_pwd)
        buttons.rejected.connect(dialog.reject)
        
        dialog.exec_()
    
    def toggle_admin_requirement(self, state):
        self.config_manager.config["require_admin"] = state == Qt.Checked
        self.config_manager.save_config()