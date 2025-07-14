import sys
import os
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QDialog
)
from ui.admin_setup_dialogs import AdminSetupDialog
from ui.admin_login import AdminLoginDialog
from ui.config_manager import ConfigManager
from ui.explorer_window import ExplorerWindow

class MainApplication:
    """Main application controller"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
         # Set logo aplikasi
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "ownize_logo_2.png")
        self.app.setWindowIcon(QIcon(logo_path))
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