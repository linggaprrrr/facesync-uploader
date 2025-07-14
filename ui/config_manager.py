import os
import json
import hashlib

class ConfigManager:
    """Mengelola konfigurasi aplikasi dan autentikasi"""
    
    def __init__(self, config_file="app_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load konfigurasi dari file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "admin_password_hash": None,
            "allowed_paths": [],
            "require_admin": True,
            "is_configured": False
        }
    
    def save_config(self):
        """Simpan konfigurasi ke file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def hash_password(self, password):
        """Hash password dengan SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password):
        """Verifikasi password admin"""
        if not self.config.get("admin_password_hash"):
            return False
        return self.hash_password(password) == self.config["admin_password_hash"]
    
    def set_admin_password(self, password):
        """Set password admin"""
        self.config["admin_password_hash"] = self.hash_password(password)
        self.config["is_configured"] = True
        return self.save_config()
    
    def add_allowed_path(self, path):
        """Tambah path yang diizinkan"""
        if path not in self.config["allowed_paths"]:
            self.config["allowed_paths"].append(path)
            return self.save_config()
        return True
    
    def remove_allowed_path(self, path):
        """Hapus path yang diizinkan"""
        if path in self.config["allowed_paths"]:
            self.config["allowed_paths"].remove(path)
            return self.save_config()
        return True
    
    def is_path_allowed(self, path):
        """Cek apakah path diizinkan"""
        if not self.config.get("require_admin", True):
            return True
        
        path = os.path.abspath(path)
        for allowed_path in self.config["allowed_paths"]:
            allowed_path = os.path.abspath(allowed_path)
            if path.startswith(allowed_path):
                return True
        return False
    
    def is_configured(self):
        """Cek apakah aplikasi sudah dikonfigurasi"""
        return self.config.get("is_configured", False)