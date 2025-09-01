"""
Enterprise Configuration Management for StudyForge AI
Provides centralized configuration with validation, environment-specific settings,
and secure credential management
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from cryptography.fernet import Fernet
import base64
import hashlib

class Environment(Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Log level configurations"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SecurityConfig:
    """Security-related configuration"""
    enable_https: bool = True
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    login_lockout_minutes: int = 15
    password_min_length: int = 12
    require_mfa: bool = False
    jwt_secret_key: Optional[str] = None
    encryption_key: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    enable_audit_logging: bool = True
    enable_ip_blocking: bool = True
    blocked_ips: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Generate secure keys if not provided
        if not self.jwt_secret_key:
            self.jwt_secret_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        if not self.encryption_key:
            self.encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "studyforge"
    username: str = ""
    password: str = ""
    ssl_mode: str = "disable"
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 30
    
@dataclass
class RedisConfig:
    """Redis configuration for caching and sessions"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    ssl: bool = False
    connection_pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

@dataclass
class AIModelConfig:
    """AI model configuration"""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "deepseek-r1:14b"
    timeout_seconds: int = 300
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    fallback_models: List[str] = field(default_factory=lambda: ["llama2:13b", "mistral:7b"])
    retry_attempts: int = 3
    retry_delay_seconds: int = 2
    enable_streaming: bool = True
    context_window: int = 8192

@dataclass
class WebSearchConfig:
    """Web search configuration"""
    enabled: bool = True
    providers: List[str] = field(default_factory=lambda: ["duckduckgo", "wikipedia"])
    max_results_per_provider: int = 5
    timeout_seconds: int = 30
    cache_ttl_hours: int = 24
    user_agent: str = "StudyForge-AI/1.0 Educational Assistant"
    max_content_length: int = 6000
    enable_content_extraction: bool = True
    rate_limit_delay: float = 1.0

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    jaeger_enabled: bool = False
    jaeger_endpoint: str = "http://localhost:14268"
    log_sampling_rate: float = 1.0
    performance_tracking: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_ms": 5000,
        "error_rate_percent": 5.0,
        "memory_usage_percent": 85.0,
        "disk_usage_percent": 90.0
    })

@dataclass
class EmailConfig:
    """Email notification configuration"""
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    from_email: str = ""
    admin_emails: List[str] = field(default_factory=list)

@dataclass
class EnterpriseConfig:
    """Main enterprise configuration container"""
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "StudyForge AI"
    app_version: str = "1.0.0"
    debug_mode: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # Component configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    
    # Runtime settings
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 300
    graceful_shutdown_timeout: int = 30
    worker_processes: int = 1
    
    def __post_init__(self):
        # Adjust settings based on environment
        if self.environment == Environment.PRODUCTION:
            self.debug_mode = False
            self.log_level = LogLevel.WARNING
            self.security.require_mfa = True
            self.security.rate_limit_requests_per_minute = 30
            self.monitoring.log_sampling_rate = 0.1
        elif self.environment == Environment.TESTING:
            self.debug_mode = True
            self.log_level = LogLevel.DEBUG
            self.database.database = "studyforge_test"

class ConfigurationManager:
    """
    Centralized configuration management with environment-specific loading,
    validation, and secure credential handling
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Determine environment
        self.environment = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
        
        # Initialize configuration
        self.config: EnterpriseConfig = None
        self._encryption_key = None
        
        # Load configuration
        self._load_configuration()
        
    def _load_configuration(self):
        """Load configuration from files and environment variables"""
        try:
            # Start with default configuration
            self.config = EnterpriseConfig(environment=self.environment)
            
            # Load base configuration file
            base_config_file = self.config_dir / "config.yaml"
            if base_config_file.exists():
                self._merge_config_file(base_config_file)
            
            # Load environment-specific configuration
            env_config_file = self.config_dir / f"config.{self.environment.value}.yaml"
            if env_config_file.exists():
                self._merge_config_file(env_config_file)
            
            # Override with environment variables
            self._load_environment_variables()
            
            # Load encrypted secrets
            self._load_encrypted_secrets()
            
            # Validate configuration
            self._validate_configuration()
            
            logging.info(f"Configuration loaded successfully for {self.environment.value} environment")
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            # Fall back to default configuration
            self.config = EnterpriseConfig(environment=self.environment)
    
    def _merge_config_file(self, config_file: Path):
        """Merge configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._deep_merge_dict(asdict(self.config), file_config)
                # Reconstruct config object from merged dict
                self.config = self._dict_to_config(asdict(self.config))
                
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    def _deep_merge_dict(self, base_dict: Dict, override_dict: Dict):
        """Deep merge two dictionaries"""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _dict_to_config(self, config_dict: Dict) -> EnterpriseConfig:
        """Convert dictionary back to EnterpriseConfig object"""
        # This is a simplified conversion - in production, you might want
        # more sophisticated deserialization
        config = EnterpriseConfig()
        
        # Update top-level attributes
        for key, value in config_dict.items():
            if hasattr(config, key) and not key.startswith('_'):
                if isinstance(getattr(config, key), (SecurityConfig, DatabaseConfig, 
                                                   RedisConfig, AIModelConfig, 
                                                   WebSearchConfig, MonitoringConfig, 
                                                   EmailConfig)):
                    # Handle nested config objects
                    continue
                setattr(config, key, value)
        
        return config
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'STUDYFORGE_DEBUG': ('debug_mode', bool),
            'STUDYFORGE_LOG_LEVEL': ('log_level', lambda x: LogLevel(x.upper())),
            'STUDYFORGE_MAX_CONCURRENT_REQUESTS': ('max_concurrent_requests', int),
            
            # Database
            'DATABASE_HOST': ('database.host', str),
            'DATABASE_PORT': ('database.port', int),
            'DATABASE_NAME': ('database.database', str),
            'DATABASE_USER': ('database.username', str),
            'DATABASE_PASSWORD': ('database.password', str),
            
            # Redis
            'REDIS_HOST': ('redis.host', str),
            'REDIS_PORT': ('redis.port', int),
            'REDIS_PASSWORD': ('redis.password', str),
            
            # AI Model
            'AI_MODEL_URL': ('ai_model.base_url', str),
            'AI_MODEL_NAME': ('ai_model.model_name', str),
            'AI_MODEL_TIMEOUT': ('ai_model.timeout_seconds', int),
            
            # Security
            'JWT_SECRET_KEY': ('security.jwt_secret_key', str),
            'ENCRYPTION_KEY': ('security.encryption_key', str),
            'SESSION_TIMEOUT': ('security.session_timeout_minutes', int),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value) if callable(converter) else value
                    self._set_nested_attribute(self.config, config_path, converted_value)
                except (ValueError, KeyError) as e:
                    logging.warning(f"Invalid environment variable {env_var}={value}: {e}")
    
    def _set_nested_attribute(self, obj, path: str, value):
        """Set nested attribute using dot notation"""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _load_encrypted_secrets(self):
        """Load encrypted secrets from secure file"""
        secrets_file = self.config_dir / "secrets.encrypted"
        if not secrets_file.exists():
            return
        
        try:
            encryption_key = self._get_encryption_key()
            if not encryption_key:
                logging.warning("No encryption key available for secrets")
                return
            
            cipher = Fernet(encryption_key)
            
            with open(secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = cipher.decrypt(encrypted_data)
            secrets = json.loads(decrypted_data.decode())
            
            # Apply secrets to configuration
            for key, value in secrets.items():
                if '.' in key:
                    self._set_nested_attribute(self.config, key, value)
                else:
                    setattr(self.config, key, value)
                    
        except Exception as e:
            logging.error(f"Failed to load encrypted secrets: {e}")
    
    def _get_encryption_key(self) -> Optional[bytes]:
        """Get or generate encryption key"""
        if self._encryption_key:
            return self._encryption_key
        
        # Try to load from environment
        key_str = os.getenv('STUDYFORGE_ENCRYPTION_KEY')
        if key_str:
            try:
                self._encryption_key = base64.urlsafe_b64decode(key_str)
                return self._encryption_key
            except Exception as e:
                logging.warning(f"Invalid encryption key from environment: {e}")
        
        # Try to load from file
        key_file = self.config_dir / ".encryption_key"
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    self._encryption_key = f.read()
                return self._encryption_key
            except Exception as e:
                logging.warning(f"Failed to load encryption key from file: {e}")
        
        # Generate new key
        self._encryption_key = Fernet.generate_key()
        
        # Save key to file (only in development)
        if self.environment == Environment.DEVELOPMENT:
            try:
                with open(key_file, 'wb') as f:
                    f.write(self._encryption_key)
                key_file.chmod(0o600)  # Restrict permissions
            except Exception as e:
                logging.warning(f"Failed to save encryption key: {e}")
        
        return self._encryption_key
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate required settings for production
        if self.config.environment == Environment.PRODUCTION:
            if not self.config.security.jwt_secret_key:
                errors.append("JWT secret key is required in production")
            
            if self.config.debug_mode:
                errors.append("Debug mode must be disabled in production")
            
            if not self.config.monitoring.enabled:
                errors.append("Monitoring must be enabled in production")
        
        # Validate database configuration
        if self.config.database.type == "postgresql":
            if not all([self.config.database.host, self.config.database.username, 
                       self.config.database.database]):
                errors.append("PostgreSQL requires host, username, and database")
        
        # Validate AI model configuration
        if not self.config.ai_model.base_url:
            errors.append("AI model base URL is required")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            if self.config.environment == Environment.PRODUCTION:
                raise ValueError(error_msg)
            else:
                logging.warning(error_msg)
    
    def get_config(self) -> EnterpriseConfig:
        """Get the current configuration"""
        return self.config
    
    def save_secrets(self, secrets: Dict[str, str]):
        """Encrypt and save secrets to secure file"""
        try:
            encryption_key = self._get_encryption_key()
            if not encryption_key:
                raise ValueError("No encryption key available")
            
            cipher = Fernet(encryption_key)
            secrets_json = json.dumps(secrets, indent=2)
            encrypted_data = cipher.encrypt(secrets_json.encode())
            
            secrets_file = self.config_dir / "secrets.encrypted"
            with open(secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            secrets_file.chmod(0o600)  # Restrict permissions
            logging.info("Secrets saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save secrets: {e}")
            raise
    
    def generate_config_template(self):
        """Generate configuration template files"""
        try:
            # Generate base config template
            template_config = asdict(EnterpriseConfig())
            
            config_file = self.config_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            
            # Generate environment-specific templates
            for env in Environment:
                env_config = asdict(EnterpriseConfig(environment=env))
                env_file = self.config_dir / f"config.{env.value}.yaml"
                with open(env_file, 'w') as f:
                    yaml.dump(env_config, f, default_flow_style=False, indent=2)
            
            logging.info("Configuration templates generated successfully")
            
        except Exception as e:
            logging.error(f"Failed to generate config templates: {e}")
            raise

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(config_dir: str = "config", environment: str = None) -> ConfigurationManager:
    """Get or create the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir, environment)
    return _config_manager

def get_config() -> EnterpriseConfig:
    """Get the current enterprise configuration"""
    return get_config_manager().get_config()