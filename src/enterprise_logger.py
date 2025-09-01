"""
Enterprise-grade logging system for StudyForge AI
Provides structured logging, audit trails, and monitoring capabilities
"""

import logging
import logging.handlers
import json
import traceback
import time
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from enum import Enum
from pathlib import Path
import threading
from contextlib import contextmanager


class LogLevel(Enum):
    """Log levels for different types of events"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60


class SecurityEventType(Enum):
    """Security event classifications"""
    AUTHENTICATION = "auth"
    AUTHORIZATION = "authz"
    DATA_ACCESS = "data_access"
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious"
    SYSTEM_BREACH = "breach"


class EnterpriseLogger:
    """
    Enterprise-grade logging system with structured logging, audit trails,
    and security event monitoring
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for consistent logging across application"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
            
        self.config = config or self._get_default_config()
        self.log_dir = Path(self.config.get('log_directory', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # Thread-local storage for request context
        self._local = threading.local()
        
        # Initialize loggers
        self._setup_loggers()
        self._initialized = True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'log_directory': 'logs',
            'log_level': 'INFO',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'format': 'json',
            'enable_console': True,
            'enable_audit': True,
            'enable_security': True,
            'compress_rotated': True
        }
    
    def _setup_loggers(self):
        """Setup enterprise logging infrastructure"""
        # Main application logger
        self.app_logger = self._create_logger(
            'studyforge.app',
            self.log_dir / 'application.log'
        )
        
        # Audit logger (cannot be disabled in production)
        self.audit_logger = self._create_logger(
            'studyforge.audit',
            self.log_dir / 'audit.log',
            level=logging.INFO
        )
        
        # Security events logger
        self.security_logger = self._create_logger(
            'studyforge.security',
            self.log_dir / 'security.log',
            level=logging.WARNING
        )
        
        # Performance monitoring logger
        self.perf_logger = self._create_logger(
            'studyforge.performance',
            self.log_dir / 'performance.log'
        )
        
        # Error tracking logger
        self.error_logger = self._create_logger(
            'studyforge.errors',
            self.log_dir / 'errors.log',
            level=logging.ERROR
        )
    
    def _create_logger(self, name: str, log_file: Path, 
                      level: int = None) -> logging.Logger:
        """Create a configured logger with rotation"""
        logger = logging.getLogger(name)
        logger.setLevel(level or getattr(logging, self.config['log_level']))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count']
        )
        
        if self.config['format'] == 'json':
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(StandardFormatter())
        
        logger.addHandler(file_handler)
        
        # Console handler for development
        if self.config['enable_console'] and not self._is_production():
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            logger.addHandler(console_handler)
        
        return logger
    
    def _is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    def set_request_context(self, request_id: str, user_id: str = None,
                           session_id: str = None, ip_address: str = None):
        """Set request context for correlation"""
        self._local.context = {
            'request_id': request_id,
            'user_id': user_id,
            'session_id': session_id,
            'ip_address': ip_address,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_context(self) -> Dict[str, Any]:
        """Get current request context"""
        return getattr(self._local, 'context', {})
    
    @contextmanager
    def request_context(self, request_id: str, **kwargs):
        """Context manager for request correlation"""
        old_context = getattr(self._local, 'context', None)
        try:
            self.set_request_context(request_id, **kwargs)
            yield
        finally:
            self._local.context = old_context
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log(self.app_logger.info, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log(self.app_logger.debug, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log(self.app_logger.warning, message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with context and stack trace"""
        extra_data = kwargs.copy()
        if error:
            extra_data.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc() if sys.exc_info()[0] else None
            })
        self._log(self.error_logger.error, message, **extra_data)
    
    def critical(self, message: str, error: Exception = None, **kwargs):
        """Log critical message with immediate alert"""
        extra_data = kwargs.copy()
        if error:
            extra_data.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            })
        self._log(self.error_logger.critical, message, **extra_data)
        
        # In production, this would trigger immediate alerts
        if self._is_production():
            self._trigger_alert(message, extra_data)
    
    def audit(self, action: str, resource: str = None, outcome: str = "success",
             **kwargs):
        """Log audit event for compliance"""
        audit_data = {
            'action': action,
            'resource': resource,
            'outcome': outcome,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._log(self.audit_logger.info, f"AUDIT: {action}", **audit_data)
    
    def security(self, event_type: SecurityEventType, message: str, 
                severity: str = "medium", **kwargs):
        """Log security event"""
        security_data = {
            'event_type': event_type.value,
            'severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._log(self.security_logger.warning, 
                 f"SECURITY: {event_type.value} - {message}", **security_data)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        perf_data = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._log(self.perf_logger.info, f"PERF: {operation}", **perf_data)
    
    def _log(self, log_func, message: str, **kwargs):
        """Internal logging with context injection"""
        context = self.get_context() or {}
        log_data = {**context, **kwargs}
        
        # Create extra dict for structured logging
        extra = {'log_data': log_data}
        log_func(message, extra=extra)
    
    def _trigger_alert(self, message: str, data: Dict[str, Any]):
        """Trigger immediate alert for critical events"""
        # In production, this would integrate with alerting systems
        # like PagerDuty, Slack, email, etc.
        alert_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': message,
            'data': data,
            'context': self.get_context()
        }
        
        # For now, write to a special alert log
        alert_file = self.log_dir / 'alerts.log'
        try:
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert_data, default=str) + '\n')
        except Exception as e:
            # Fallback to stderr if alert logging fails
            print(f"CRITICAL ALERT LOGGING FAILED: {e}", file=sys.stderr)
            print(json.dumps(alert_data, default=str), file=sys.stderr)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Add structured data if present
        if hasattr(record, 'log_data'):
            log_entry['data'] = record.log_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StandardFormatter(logging.Formatter):
    """Standard text formatter with context"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        return f"{color}[{timestamp}] {record.levelname:8s} {record.name}: {record.getMessage()}{reset}"


# Performance monitoring decorator
def log_performance(operation_name: str = None, logger: EnterpriseLogger = None):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = EnterpriseLogger()
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance(name, duration, status='success')
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.performance(name, duration, status='error', error=str(e))
                raise
        
        return wrapper
    return decorator


# Global logger instance
logger = EnterpriseLogger()