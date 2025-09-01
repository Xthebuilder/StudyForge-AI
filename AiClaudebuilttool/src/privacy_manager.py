"""
Privacy and Security Manager for StudyForge AI Enhanced Edition
Handles encryption, data anonymization, and privacy controls
"""

import os
import json
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Try to import cryptography for encryption features
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ cryptography not installed. Install with: pip install cryptography")

@dataclass
class PrivacySettings:
    local_only_mode: bool = False
    encrypt_conversations: bool = False
    anonymize_analytics: bool = True
    data_retention_days: int = 365
    share_learning_patterns: bool = False
    export_allowed: bool = True
    delete_on_request: bool = True

class PrivacyManager:
    """Comprehensive privacy and security management"""
    
    def __init__(self, enable_encryption: bool = False, local_only_mode: bool = False):
        self.logger = logging.getLogger(f"{__name__}.PrivacyManager")
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        self.local_only_mode = local_only_mode
        
        # Encryption setup
        self.encryption_key = None
        if self.enable_encryption:
            self._setup_encryption()
        
        # Privacy settings database
        self.privacy_db_path = "privacy_settings.db"
        self._init_privacy_database()
        
        print(f"🔒 Privacy Manager initialized")
        print(f"   Encryption: {'Enabled' if self.enable_encryption else 'Disabled'}")
        print(f"   Local-only mode: {'Enabled' if self.local_only_mode else 'Disabled'}")
    
    def _setup_encryption(self):
        """Setup encryption key and cipher"""
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Encryption requested but cryptography library not available")
            return
        
        try:
            # Check for existing key
            key_file = "encryption.key"
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                os.chmod(key_file, 0o600)  # Restrict permissions
            
            # Test encryption
            cipher = Fernet(self.encryption_key)
            test_data = b"test"
            encrypted = cipher.encrypt(test_data)
            decrypted = cipher.decrypt(encrypted)
            assert decrypted == test_data
            
            self.logger.info("Encryption setup successful")
            
        except Exception as e:
            self.logger.error(f"Failed to setup encryption: {e}")
            self.enable_encryption = False
    
    def _init_privacy_database(self):
        """Initialize privacy settings database"""
        conn = sqlite3.connect(self.privacy_db_path)
        conn.row_factory = sqlite3.Row
        
        # User privacy settings
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_privacy_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                settings_json TEXT NOT NULL,
                consent_given BOOLEAN DEFAULT FALSE,
                consent_date REAL,
                last_updated REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Data access log
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                access_type TEXT NOT NULL,
                data_category TEXT NOT NULL,
                purpose TEXT NOT NULL,
                timestamp REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Data export requests
        conn.execute('''
            CREATE TABLE IF NOT EXISTS export_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                request_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                file_path TEXT,
                created_at REAL DEFAULT (julianday('now')),
                completed_at REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.enable_encryption or not self.encryption_key:
            return data
        
        try:
            cipher = Fernet(self.encryption_key)
            encrypted_bytes = cipher.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.enable_encryption or not self.encryption_key:
            return encrypted_data
        
        try:
            cipher = Fernet(self.encryption_key)
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def anonymize_text(self, text: str, user_id: str) -> str:
        """Anonymize personal information in text"""
        # Create consistent but anonymous user identifier
        anonymous_id = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        
        # Replace potential personal information
        import re
        
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     f'user_{anonymous_id}@example.com', text)
        
        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'XXX-XXX-XXXX', text)
        
        # Names (simple heuristic - capitalized words)
        # Be conservative to avoid removing legitimate technical terms
        text = re.sub(r'\bMy name is [A-Z][a-z]+\b', f'My name is User{anonymous_id}', text)
        text = re.sub(r'\bI am [A-Z][a-z]+\b', f'I am User{anonymous_id}', text)
        
        return text
    
    def get_user_privacy_settings(self, user_id: str) -> PrivacySettings:
        """Get privacy settings for a user"""
        try:
            conn = sqlite3.connect(self.privacy_db_path)
            conn.row_factory = sqlite3.Row
            
            row = conn.execute(
                'SELECT settings_json FROM user_privacy_settings WHERE user_id = ?',
                (user_id,)
            ).fetchone()
            
            conn.close()
            
            if row:
                settings_dict = json.loads(row['settings_json'])
                return PrivacySettings(**settings_dict)
            else:
                # Return default settings
                return PrivacySettings()
                
        except Exception as e:
            self.logger.error(f"Failed to get privacy settings: {e}")
            return PrivacySettings()
    
    def update_privacy_settings(self, user_id: str, settings: PrivacySettings) -> bool:
        """Update privacy settings for a user"""
        try:
            settings_json = json.dumps(settings.__dict__)
            
            conn = sqlite3.connect(self.privacy_db_path)
            conn.execute('''
                INSERT OR REPLACE INTO user_privacy_settings 
                (user_id, settings_json, consent_given, consent_date, last_updated)
                VALUES (?, ?, ?, julianday('now'), julianday('now'))
            ''', (user_id, settings_json, True))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update privacy settings: {e}")
            return False
    
    def log_data_access(self, user_id: str, access_type: str, data_category: str, purpose: str):
        """Log data access for transparency"""
        try:
            conn = sqlite3.connect(self.privacy_db_path)
            conn.execute('''
                INSERT INTO data_access_log 
                (user_id, access_type, data_category, purpose)
                VALUES (?, ?, ?, ?)
            ''', (user_id, access_type, data_category, purpose))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log data access: {e}")
    
    def can_use_web_features(self, user_id: str) -> bool:
        """Check if web features are allowed for user"""
        if self.local_only_mode:
            return False
        
        settings = self.get_user_privacy_settings(user_id)
        return not settings.local_only_mode
    
    def can_store_conversation(self, user_id: str) -> bool:
        """Check if conversation storage is allowed"""
        settings = self.get_user_privacy_settings(user_id)
        return True  # Always allow for core functionality
    
    def should_encrypt_conversation(self, user_id: str) -> bool:
        """Check if conversation should be encrypted"""
        if not self.enable_encryption:
            return False
        
        settings = self.get_user_privacy_settings(user_id)
        return settings.encrypt_conversations
    
    def clean_expired_data(self, user_id: str = None):
        """Clean expired data based on retention policy"""
        try:
            if user_id:
                settings = self.get_user_privacy_settings(user_id)
                retention_days = settings.data_retention_days
                users_to_clean = [user_id]
            else:
                retention_days = 365  # Default
                # Get all users (would need access to main database)
                users_to_clean = []  # Simplified for now
            
            cutoff_timestamp = (datetime.now().timestamp() - (retention_days * 24 * 3600))
            
            # Clean privacy database
            conn = sqlite3.connect(self.privacy_db_path)
            
            if users_to_clean:
                placeholders = ','.join(['?' for _ in users_to_clean])
                params = users_to_clean + [cutoff_timestamp]
                
                conn.execute(f'''
                    DELETE FROM data_access_log 
                    WHERE user_id IN ({placeholders}) AND timestamp < ?
                ''', params)
            else:
                conn.execute('DELETE FROM data_access_log WHERE timestamp < ?', 
                           (cutoff_timestamp,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned expired data for retention period: {retention_days} days")
            
        except Exception as e:
            self.logger.error(f"Failed to clean expired data: {e}")
    
    def export_user_data(self, user_id: str, export_format: str = 'json') -> Optional[str]:
        """Export all user data for GDPR compliance"""
        try:
            export_data = {
                'user_id': user_id,
                'export_date': datetime.now().isoformat(),
                'privacy_settings': self.get_user_privacy_settings(user_id).__dict__,
                'data_access_log': [],
                'conversations': [],  # Would be populated from main database
                'learning_progress': [],  # Would be populated from learning tracker
                'study_sessions': []  # Would be populated from session manager
            }
            
            # Get data access log
            conn = sqlite3.connect(self.privacy_db_path)
            conn.row_factory = sqlite3.Row
            
            access_logs = conn.execute('''
                SELECT access_type, data_category, purpose, timestamp
                FROM data_access_log 
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id,)).fetchall()
            
            export_data['data_access_log'] = [
                {
                    'access_type': log['access_type'],
                    'data_category': log['data_category'],
                    'purpose': log['purpose'],
                    'timestamp': datetime.fromtimestamp(log['timestamp']).isoformat()
                }
                for log in access_logs
            ]
            
            conn.close()
            
            # Create export file
            export_filename = f"user_data_export_{user_id}_{int(datetime.now().timestamp())}.json"
            export_path = f"exports/{export_filename}"
            
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Log export request
            conn = sqlite3.connect(self.privacy_db_path)
            conn.execute('''
                INSERT INTO export_requests 
                (user_id, request_type, status, file_path, completed_at)
                VALUES (?, ?, ?, ?, julianday('now'))
            ''', (user_id, export_format, 'completed', export_path))
            conn.commit()
            conn.close()
            
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export user data: {e}")
            return None
    
    def delete_user_data(self, user_id: str, confirmation_code: str) -> bool:
        """Delete all user data (GDPR right to be forgotten)"""
        # In a real implementation, this would require proper confirmation
        # and would delete data across all databases
        
        expected_code = hashlib.sha256(f"delete_{user_id}".encode()).hexdigest()[:8]
        
        if confirmation_code != expected_code:
            self.logger.warning(f"Invalid deletion confirmation code for user {user_id}")
            return False
        
        try:
            # Delete from privacy database
            conn = sqlite3.connect(self.privacy_db_path)
            conn.execute('DELETE FROM user_privacy_settings WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM data_access_log WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM export_requests WHERE user_id = ?', (user_id,))
            conn.commit()
            conn.close()
            
            # Note: In a real implementation, this would also delete from:
            # - Main conversation database
            # - Learning progress database
            # - Study session database
            # - Semantic memory database
            # - Any cached files or exports
            
            self.logger.info(f"User data deleted for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete user data: {e}")
            return False
    
    def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy transparency report"""
        try:
            settings = self.get_user_privacy_settings(user_id)
            
            conn = sqlite3.connect(self.privacy_db_path)
            conn.row_factory = sqlite3.Row
            
            # Data access statistics
            access_stats = conn.execute('''
                SELECT data_category, COUNT(*) as access_count,
                       MAX(timestamp) as last_access
                FROM data_access_log 
                WHERE user_id = ?
                GROUP BY data_category
            ''', (user_id,)).fetchall()
            
            # Export history
            exports = conn.execute('''
                SELECT request_type, status, created_at, completed_at
                FROM export_requests 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,)).fetchall()
            
            conn.close()
            
            report = {
                'user_id': user_id,
                'report_date': datetime.now().isoformat(),
                'privacy_settings': settings.__dict__,
                'data_categories': [
                    {
                        'category': stat['data_category'],
                        'access_count': stat['access_count'],
                        'last_access': datetime.fromtimestamp(stat['last_access']).isoformat() if stat['last_access'] else None
                    }
                    for stat in access_stats
                ],
                'export_history': [
                    {
                        'type': exp['request_type'],
                        'status': exp['status'],
                        'requested': datetime.fromtimestamp(exp['created_at']).isoformat(),
                        'completed': datetime.fromtimestamp(exp['completed_at']).isoformat() if exp['completed_at'] else None
                    }
                    for exp in exports
                ],
                'encryption_enabled': self.enable_encryption,
                'local_only_mode': self.local_only_mode
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate privacy report: {e}")
            return {}
    
    def get_deletion_confirmation_code(self, user_id: str) -> str:
        """Get confirmation code for account deletion"""
        return hashlib.sha256(f"delete_{user_id}".encode()).hexdigest()[:8]