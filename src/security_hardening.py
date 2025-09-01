"""
Security Hardening Module for StudyForge AI
Implements comprehensive security measures including input validation,
authentication, authorization, rate limiting, and threat detection
"""

import re
import json
import hmac
import hashlib
import secrets
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import bleach
import html
from urllib.parse import urlparse
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import bcrypt
import jwt
from .enterprise_logger import EnterpriseLogger, SecurityEventType

class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """Types of security attacks"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALICIOUS_FILE = "malicious_file"

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_file_size: int = 50 * 1024 * 1024     # 50MB
    allowed_file_types: List[str] = field(default_factory=lambda: [
        'pdf', 'txt', 'md', 'docx', 'doc', 'html', 'json', 'csv', 'xml'
    ])
    max_query_length: int = 10000
    max_filename_length: int = 255
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 60
    session_timeout: int = 1800  # 30 minutes
    password_min_length: int = 12
    jwt_expiration: int = 3600  # 1 hour
    enable_ip_blocking: bool = True
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes

@dataclass
class ThreatEvent:
    """Security threat event"""
    threat_type: AttackType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    payload: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = False

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = EnterpriseLogger()
        
        # Compile regex patterns for efficiency
        self._sql_patterns = [
            re.compile(r'\b(union|select|insert|update|delete|drop|create|alter|exec)\b', re.IGNORECASE),
            re.compile(r'[\'";].*[\'";]', re.IGNORECASE),
            re.compile(r'--.*$', re.MULTILINE),
            re.compile(r'/\*.*?\*/', re.DOTALL)
        ]
        
        self._xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            re.compile(r'<object[^>]*>', re.IGNORECASE)
        ]
        
        self._path_traversal_patterns = [
            re.compile(r'\.\./'),
            re.compile(r'\.\.\\\\'),
            re.compile(r'/etc/passwd'),
            re.compile(r'\\\\windows\\\\system32')
        ]
        
        self._command_injection_patterns = [
            re.compile(r'[;&|`$]'),
            re.compile(r'\$\(.*\)'),
            re.compile(r'`.*`'),
            re.compile(r'>\s*/dev/null')
        ]
    
    def validate_query(self, query: str, user_ip: str, user_id: str = None) -> Tuple[bool, Optional[ThreatEvent]]:
        """Validate user query for security threats"""
        if not query or len(query) > self.config.max_query_length:
            threat = ThreatEvent(
                threat_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=user_ip,
                user_id=user_id,
                description=f"Query length validation failed: {len(query) if query else 0} chars",
                payload=query[:500] if query else None
            )
            return False, threat
        
        # Check for SQL injection patterns
        for pattern in self._sql_patterns:
            if pattern.search(query):
                threat = ThreatEvent(
                    threat_type=AttackType.SQL_INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=user_ip,
                    user_id=user_id,
                    description="SQL injection pattern detected",
                    payload=pattern.pattern
                )
                return False, threat
        
        # Check for XSS patterns
        for pattern in self._xss_patterns:
            if pattern.search(query):
                threat = ThreatEvent(
                    threat_type=AttackType.XSS,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=user_ip,
                    user_id=user_id,
                    description="XSS pattern detected",
                    payload=pattern.pattern
                )
                return False, threat
        
        # Check for command injection
        for pattern in self._command_injection_patterns:
            if pattern.search(query):
                threat = ThreatEvent(
                    threat_type=AttackType.COMMAND_INJECTION,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip=user_ip,
                    user_id=user_id,
                    description="Command injection pattern detected",
                    payload=pattern.pattern
                )
                return False, threat
        
        return True, None
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text input to prevent XSS"""
        if not text:
            return ""
        
        # HTML escape
        text = html.escape(text)
        
        # Remove potentially dangerous HTML tags
        allowed_tags = ['b', 'i', 'em', 'strong', 'p', 'br']
        text = bleach.clean(text, tags=allowed_tags, strip=True)
        
        return text
    
    def validate_filename(self, filename: str, user_ip: str, user_id: str = None) -> Tuple[bool, Optional[ThreatEvent]]:
        """Validate uploaded filename for security"""
        if not filename or len(filename) > self.config.max_filename_length:
            threat = ThreatEvent(
                threat_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=user_ip,
                user_id=user_id,
                description=f"Filename validation failed: {filename}",
                payload=filename
            )
            return False, threat
        
        # Check for path traversal
        for pattern in self._path_traversal_patterns:
            if pattern.search(filename):
                threat = ThreatEvent(
                    threat_type=AttackType.PATH_TRAVERSAL,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=user_ip,
                    user_id=user_id,
                    description="Path traversal detected in filename",
                    payload=filename
                )
                return False, threat
        
        # Check file extension
        if '.' in filename:
            extension = filename.split('.')[-1].lower()
            if extension not in self.config.allowed_file_types:
                threat = ThreatEvent(
                    threat_type=AttackType.MALICIOUS_FILE,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=user_ip,
                    user_id=user_id,
                    description=f"Disallowed file type: {extension}",
                    payload=filename
                )
                return False, threat
        
        return True, None
    
    def validate_file_content(self, content: bytes, filename: str, user_ip: str, user_id: str = None) -> Tuple[bool, Optional[ThreatEvent]]:
        """Validate file content for malicious patterns"""
        if len(content) > self.config.max_file_size:
            threat = ThreatEvent(
                threat_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=user_ip,
                user_id=user_id,
                description=f"File size exceeded: {len(content)} bytes",
                payload=filename
            )
            return False, threat
        
        # Check for common malicious patterns in binary content
        malicious_signatures = [
            b'MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
            b'<?php',  # PHP script
            b'<script>',  # JavaScript
            b'eval(',  # Evaluation functions
            b'system(',  # System calls
            b'exec(',  # Execution functions
        ]
        
        content_lower = content.lower()
        for signature in malicious_signatures:
            if signature in content_lower:
                threat = ThreatEvent(
                    threat_type=AttackType.MALICIOUS_FILE,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=user_ip,
                    user_id=user_id,
                    description=f"Malicious signature detected: {signature.decode('utf-8', errors='ignore')}",
                    payload=filename
                )
                return False, threat
        
        return True, None

class RateLimiter:
    """Request rate limiting with sliding window"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock timestamp
        self.logger = EnterpriseLogger()
    
    def check_rate_limit(self, identifier: str, user_ip: str) -> Tuple[bool, Optional[ThreatEvent]]:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Check if IP is currently blocked
        if user_ip in self.blocked_ips:
            if current_time < self.blocked_ips[user_ip]:
                threat = ThreatEvent(
                    threat_type=AttackType.RATE_LIMIT_EXCEEDED,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=user_ip,
                    user_id=identifier,
                    description="IP is currently blocked",
                    payload=None,
                    blocked=True
                )
                return False, threat
            else:
                # Unblock IP
                del self.blocked_ips[user_ip]
        
        # Initialize request history if needed
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside the window
        window_start = current_time - self.config.rate_limit_window
        self.requests[identifier] = [req_time for req_time in self.requests[identifier] if req_time > window_start]
        
        # Check if rate limit exceeded
        if len(self.requests[identifier]) >= self.config.rate_limit_max_requests:
            # Block IP temporarily
            self.blocked_ips[user_ip] = current_time + self.config.lockout_duration
            
            threat = ThreatEvent(
                threat_type=AttackType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.HIGH,
                source_ip=user_ip,
                user_id=identifier,
                description=f"Rate limit exceeded: {len(self.requests[identifier])} requests in {self.config.rate_limit_window}s",
                payload=None,
                blocked=True
            )
            return False, threat
        
        # Record this request
        self.requests[identifier].append(current_time)
        return True, None

class AuthenticationManager:
    """Secure authentication and session management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.failed_attempts: Dict[str, List[float]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.logger = EnterpriseLogger()
        
        # Generate JWT secret if not provided
        self.jwt_secret = secrets.token_urlsafe(32)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def check_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Check password strength"""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain uppercase letters")
        
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain lowercase letters")
        
        if not re.search(r'\d', password):
            issues.append("Password must contain numbers")
        
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
            issues.append("Password must contain special characters")
        
        return len(issues) == 0, issues
    
    def check_brute_force(self, identifier: str, user_ip: str) -> Tuple[bool, Optional[ThreatEvent]]:
        """Check for brute force attacks"""
        current_time = time.time()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        # Remove old attempts
        window_start = current_time - self.config.lockout_duration
        self.failed_attempts[identifier] = [
            attempt_time for attempt_time in self.failed_attempts[identifier] 
            if attempt_time > window_start
        ]
        
        # Check if max attempts exceeded
        if len(self.failed_attempts[identifier]) >= self.config.max_failed_attempts:
            threat = ThreatEvent(
                threat_type=AttackType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                source_ip=user_ip,
                user_id=identifier,
                description=f"Brute force detected: {len(self.failed_attempts[identifier])} failed attempts",
                payload=None,
                blocked=True
            )
            return False, threat
        
        return True, None
    
    def record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt"""
        current_time = time.time()
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        self.failed_attempts[identifier].append(current_time)
    
    def create_session_token(self, user_id: str, user_ip: str, additional_claims: Dict = None) -> str:
        """Create secure JWT session token"""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'ip_address': user_ip,
            'iat': now,
            'exp': now + timedelta(seconds=self.config.jwt_expiration),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # Store session
        self.active_sessions[payload['jti']] = {
            'user_id': user_id,
            'ip_address': user_ip,
            'created_at': now,
            'last_activity': now
        }
        
        self.logger.audit('session_created', 'authentication_manager',
                         user_id=user_id, ip_address=user_ip, token_id=payload['jti'])
        
        return token
    
    def validate_session_token(self, token: str, user_ip: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[ThreatEvent]]:
        """Validate session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session is active
            token_id = payload.get('jti')
            if token_id not in self.active_sessions:
                threat = ThreatEvent(
                    threat_type=AttackType.SUSPICIOUS_PATTERN,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=user_ip,
                    user_id=payload.get('user_id'),
                    description="Invalid session token",
                    payload=token[:50]
                )
                return False, None, threat
            
            # Check IP address consistency
            if payload.get('ip_address') != user_ip:
                self.revoke_session(token_id)
                threat = ThreatEvent(
                    threat_type=AttackType.SUSPICIOUS_PATTERN,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=user_ip,
                    user_id=payload.get('user_id'),
                    description="IP address mismatch for session",
                    payload=f"Expected: {payload.get('ip_address')}, Got: {user_ip}"
                )
                return False, None, threat
            
            # Update last activity
            self.active_sessions[token_id]['last_activity'] = datetime.utcnow()
            
            return True, payload, None
            
        except jwt.ExpiredSignatureError:
            threat = ThreatEvent(
                threat_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.LOW,
                source_ip=user_ip,
                user_id=None,
                description="Expired session token",
                payload=token[:50]
            )
            return False, None, threat
        except jwt.InvalidTokenError:
            threat = ThreatEvent(
                threat_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=user_ip,
                user_id=None,
                description="Invalid session token format",
                payload=token[:50]
            )
            return False, None, threat
    
    def revoke_session(self, token_id: str):
        """Revoke a specific session"""
        if token_id in self.active_sessions:
            user_id = self.active_sessions[token_id]['user_id']
            del self.active_sessions[token_id]
            self.logger.audit('session_revoked', 'authentication_manager',
                             user_id=user_id, token_id=token_id)

class SecurityHardening:
    """Main security hardening orchestrator"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.auth_manager = AuthenticationManager(self.config)
        self.logger = EnterpriseLogger()
        
        # Thread-safe threat tracking
        self.recent_threats: List[ThreatEvent] = []
        self.blocked_ips: set = set()
    
    def validate_request(self, 
                        query: str = None, 
                        filename: str = None, 
                        file_content: bytes = None,
                        user_ip: str = "127.0.0.1", 
                        user_id: str = None,
                        session_token: str = None) -> Tuple[bool, List[ThreatEvent]]:
        """Comprehensive request validation"""
        threats = []
        
        # Rate limiting check
        identifier = user_id or user_ip
        rate_ok, rate_threat = self.rate_limiter.check_rate_limit(identifier, user_ip)
        if not rate_ok:
            threats.append(rate_threat)
            self._handle_threat(rate_threat)
        
        # Session validation
        if session_token:
            session_ok, session_data, session_threat = self.auth_manager.validate_session_token(session_token, user_ip)
            if not session_ok:
                threats.append(session_threat)
                self._handle_threat(session_threat)
        
        # Query validation
        if query:
            query_ok, query_threat = self.validator.validate_query(query, user_ip, user_id)
            if not query_ok:
                threats.append(query_threat)
                self._handle_threat(query_threat)
        
        # File validation
        if filename:
            filename_ok, filename_threat = self.validator.validate_filename(filename, user_ip, user_id)
            if not filename_ok:
                threats.append(filename_threat)
                self._handle_threat(filename_threat)
        
        if file_content:
            content_ok, content_threat = self.validator.validate_file_content(file_content, filename or "unknown", user_ip, user_id)
            if not content_ok:
                threats.append(content_threat)
                self._handle_threat(content_threat)
        
        # Check if any critical threats were found
        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
        if critical_threats:
            self._block_ip(user_ip, "Critical security threat detected")
        
        return len(threats) == 0, threats
    
    def _handle_threat(self, threat: ThreatEvent):
        """Handle detected security threat"""
        self.recent_threats.append(threat)
        
        # Log security event
        self.logger.security(
            event_type=self._attack_type_to_security_event(threat.threat_type),
            message=threat.description,
            severity=threat.threat_level.value,
            source_ip=threat.source_ip,
            user_id=threat.user_id,
            payload=threat.payload,
            blocked=threat.blocked
        )
        
        # Handle based on threat level
        if threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            if threat.threat_type in [AttackType.BRUTE_FORCE, AttackType.RATE_LIMIT_EXCEEDED]:
                self._block_ip(threat.source_ip, f"{threat.threat_type.value} detected")
        
        # Keep only recent threats (last 1000)
        if len(self.recent_threats) > 1000:
            self.recent_threats = self.recent_threats[-1000:]
    
    def _attack_type_to_security_event(self, attack_type: AttackType) -> SecurityEventType:
        """Map attack type to security event type"""
        mapping = {
            AttackType.BRUTE_FORCE: SecurityEventType.AUTHENTICATION,
            AttackType.RATE_LIMIT_EXCEEDED: SecurityEventType.RATE_LIMIT,
            AttackType.SQL_INJECTION: SecurityEventType.SUSPICIOUS_ACTIVITY,
            AttackType.XSS: SecurityEventType.SUSPICIOUS_ACTIVITY,
            AttackType.COMMAND_INJECTION: SecurityEventType.SYSTEM_BREACH,
            AttackType.PATH_TRAVERSAL: SecurityEventType.DATA_ACCESS,
            AttackType.MALICIOUS_FILE: SecurityEventType.SUSPICIOUS_ACTIVITY,
        }
        return mapping.get(attack_type, SecurityEventType.SUSPICIOUS_ACTIVITY)
    
    def _block_ip(self, ip_address: str, reason: str):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        self.logger.security(
            event_type=SecurityEventType.AUTHORIZATION,
            message=f"IP blocked: {reason}",
            severity="high",
            source_ip=ip_address,
            blocked=True
        )
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.logger.audit('ip_unblocked', 'security_hardening', ip_address=ip_address)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics"""
        threat_counts = {}
        for threat in self.recent_threats:
            threat_type = threat.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            'total_threats': len(self.recent_threats),
            'blocked_ips': len(self.blocked_ips),
            'threat_breakdown': threat_counts,
            'critical_threats': len([t for t in self.recent_threats if t.threat_level == ThreatLevel.CRITICAL]),
            'recent_threats_24h': len([
                t for t in self.recent_threats 
                if (datetime.now() - t.timestamp).total_seconds() < 86400
            ])
        }

# Global security hardening instance
_security_hardening: Optional[SecurityHardening] = None

def get_security_hardening(config: SecurityConfig = None) -> SecurityHardening:
    """Get or create the global security hardening instance"""
    global _security_hardening
    if _security_hardening is None:
        _security_hardening = SecurityHardening(config)
    return _security_hardening