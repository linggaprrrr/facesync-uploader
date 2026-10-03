import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QDialog, QSplashScreen
)
# ponytail: ui.* imports are deferred to MainApplication.__init__ so the
# splash can paint before cv2/numpy/aiohttp/etc. are loaded.

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
        # ponytail: small assets on purpose — ownize_logo*.png are 11810px and
        # take ~550 MB + seconds to decode just for an icon.
        assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        self.app.setWindowIcon(QIcon(os.path.join(assets, "ownize_logo.ico")))

        # Loader: paint before any heavy import
        self.splash = QSplashScreen(
            QPixmap(os.path.join(assets, "splash.png")), Qt.WindowStaysOnTopHint
        )
        self.splash.show()
        self._status("Memuat model wajah...")

        # Face model loads in the background (CUDA init is the slow part)
        from core.device_setup import warmup_async
        warmup_async()

        self._status("Memuat antarmuka...")
        global AdminSetupDialog, AdminLoginDialog, ExplorerWindow
        from ui.admin_setup_dialogs import AdminSetupDialog
        from ui.admin_login import AdminLoginDialog
        from ui.config_manager import ConfigManager
        from ui.explorer_window import ExplorerWindow

        self.config_manager = ConfigManager()
        self.main_window = None
        self.face_detection_initialized = False
        
    def _status(self, text):
        self.splash.showMessage(text, Qt.AlignBottom | Qt.AlignHCenter, Qt.darkGray)
        self.app.processEvents()

    def _close_splash(self):
        if self.splash:
            self.splash.close()
            self.splash = None

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
            self._close_splash()
            
            # Close main window if open
            if self.main_window:
                logger.info("🔄 Closing main window...")
                self.main_window.close()
            
            logger.info("✅ Application cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def run(self):
        """Run the application with authentication flow and proper initialization"""
        
        try:
            # STEP 1: Initialize all systems FIRST
            if not self.initialize_systems():
                return 1  # Exit with error code

            self._close_splash()  # dialogs below take over from the loader
            
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