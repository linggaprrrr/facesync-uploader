
from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QLineEdit, QMessageBox, QDialog, QFormLayout,
    QDialogButtonBox
)


class AdminLoginDialog(QDialog):
    """Dialog untuk login admin"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("Login Admin - File Manager")
        self.setModal(True)
        self.setFixedSize(300, 150)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Masuk sebagai Admin")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        form_layout = QFormLayout()
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.accept)
        form_layout.addRow("Password:", self.password_input)
        
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.password_input.setFocus()
    
    def accept(self):
        password = self.password_input.text()
        
        if self.config_manager.verify_password(password):
            super().accept()
        else:
            QMessageBox.warning(self, "Error", "Password salah!")
            self.password_input.clear()
            self.password_input.setFocus()
