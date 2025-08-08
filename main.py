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

# UPDATED: Import your optimized face detection module

import logging

# Setup logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MainApplication:
    """Main application controller with optimized face detection"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # Set logo aplikasi
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "ownize_logo_2.png")
        self.app.setWindowIcon(QIcon(logo_path))
        
        self.config_manager = ConfigManager()
        self.main_window = None
        self.face_detection_initialized = False
        
    def initialize_systems(self):
        """Initialize all application systems"""
        try:
            logger.info("🚀 Starting FaceSync application initialization...")
                       
            # Example: Initialize other components, check licenses, etc.
            
            logger.info("✅ All systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            QMessageBox.critical(
                None,
                "Critical Error",
                f"System initialization failed:\n{str(e)}\n\nApplication will exit."
            )
            return False
    
    def cleanup_systems(self):
        """Cleanup all application systems"""
        try:
            logger.info("🔄 Starting application cleanup...")
            
            # Close main window if open
            if self.main_window:
                logger.info("🔄 Closing main window...")
                self.main_window.close()
            
            # Cleanup face detection system
            if self.face_detection_initialized:
                logger.info("🔄 Cleaning up face detection system...")
                cleanup_optimized_face_detection()
                self.face_detection_initialized = False
            
            logger.info("✅ Application cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def run(self):
        """Run the application with authentication flow and proper initialization"""
        
        try:
            # STEP 1: Initialize all systems FIRST
            if not self.initialize_systems():
                return 1  # Exit with error code
            
            # STEP 2: Check if app is configured
            if not self.config_manager.is_configured():
                logger.info("🔧 First time setup required")
                # First time setup
                setup_dialog = AdminSetupDialog(self.config_manager)
                if setup_dialog.exec_() != QDialog.Accepted:
                    QMessageBox.information(None, "Info", "Setup dibatalkan. Aplikasi akan keluar.")
                    return 0
                
                logger.info("✅ Initial setup completed")
            
            # STEP 3: Check if admin authentication is required
            if self.config_manager.config.get("require_admin", True):
                logger.info("🔐 Admin authentication required")
                # Show login dialog
                login_dialog = AdminLoginDialog(self.config_manager)
                if login_dialog.exec_() != QDialog.Accepted:
                    QMessageBox.information(None, "Info", "Login diperlukan untuk menggunakan aplikasi.")
                    return 0
                
                logger.info("✅ Admin authentication successful")
            
            # STEP 4: Launch main application
            logger.info("🚀 Launching main application window...")
            self.main_window = ExplorerWindow(self.config_manager)
            self.main_window.show()
            
            logger.info("✅ FaceSync application started successfully")
            
            # STEP 5: Run the application event loop
            exit_code = self.app.exec_()
            
            logger.info(f"🔄 Application exiting with code: {exit_code}")
            return exit_code
            
        except Exception as e:
            logger.error(f"❌ Application run error: {e}")
            QMessageBox.critical(
                None,
                "Application Error",
                f"Application encountered an error:\n{str(e)}"
            )
            return 1
            
        finally:
            # STEP 6: Always cleanup, regardless of how we exit
            self.cleanup_systems()


def main():
    """Main entry point with exception handling"""
    try:
        # Create and run application
        app = MainApplication()
        exit_code = app.run()
        
        # Exit with the code returned by the application
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("🔄 Application interrupted by user (Ctrl+C)")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Fatal application error: {e}")
        # Show error message if possible
        try:
            QMessageBox.critical(
                None,
                "Fatal Error",
                f"A fatal error occurred:\n{str(e)}\n\nApplication will exit."
            )
        except:
            pass  # GUI might not be available
        
        sys.exit(1)


if __name__ == '__main__':
    main()