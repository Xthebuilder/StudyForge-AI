"""
API Middleware for StudyForge AI
Provides comprehensive request validation, rate limiting, authentication,
and security middleware for the FastAPI application
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
import uvicorn
from contextlib import asynccontextmanager

from .enterprise_logger import EnterpriseLogger, SecurityEventType
from .security_hardening import SecurityHardening, SecurityConfig, ThreatLevel, AttackType
from .enterprise_config import get_config

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and threat detection"""
    
    def __init__(self, app: FastAPI, security_config: SecurityConfig = None):
        super().__init__(app)
        self.security = SecurityHardening(security_config)
        self.logger = EnterpriseLogger()
        
        # Exempt paths from security checks
        self.exempt_paths = {'/health', '/metrics', '/docs', '/openapi.json'}
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through security validation"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Set request context for logging
        request_id = f"req_{int(time.time() * 1000)}_{id(request)}"
        with self.logger.request_context(request_id, ip_address=client_ip):
            
            # Skip security checks for exempt paths
            if request.url.path in self.exempt_paths:
                response = await call_next(request)
                return self._add_security_headers(response)
            
            try:
                # Check if IP is blocked
                if self.security.is_ip_blocked(client_ip):
                    self.logger.security(
                        SecurityEventType.AUTHORIZATION,
                        "Blocked IP attempted access",
                        severity="high",
                        source_ip=client_ip
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                # Validate request size
                if hasattr(request, 'headers') and 'content-length' in request.headers:
                    content_length = int(request.headers.get('content-length', 0))
                    max_size = self.security.config.max_request_size
                    if content_length > max_size:
                        self.logger.security(
                            SecurityEventType.SUSPICIOUS_ACTIVITY,
                            f"Request size exceeded: {content_length} bytes",
                            severity="medium",
                            source_ip=client_ip
                        )
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="Request too large"
                        )
                
                # Rate limiting check
                rate_ok, threats = self.security.validate_request(
                    user_ip=client_ip,
                    user_id=self._get_user_id_from_request(request)
                )
                
                if not rate_ok:
                    # Find rate limit specific threats
                    rate_threats = [t for t in threats if t.threat_type == AttackType.RATE_LIMIT_EXCEEDED]
                    if rate_threats:
                        self.logger.security(
                            SecurityEventType.RATE_LIMIT,
                            "Rate limit exceeded",
                            severity="medium",
                            source_ip=client_ip
                        )
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded",
                            headers={"Retry-After": "60"}
                        )
                
                # Process request
                response = await call_next(request)
                
                # Log successful request
                processing_time = time.time() - start_time
                self.logger.audit(
                    'api_request_completed',
                    'api_middleware',
                    outcome='success',
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    processing_time=processing_time,
                    client_ip=client_ip
                )
                
                return self._add_security_headers(response)
                
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                # Log unexpected errors
                self.logger.error(
                    "Unexpected error in security middleware",
                    error=e,
                    client_ip=client_ip,
                    path=request.url.path
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "127.0.0.1"
    
    def _get_user_id_from_request(self, request: Request) -> Optional[str]:
        """Extract user ID from request (from session, JWT, etc.)"""
        # This would integrate with your authentication system
        # For now, return None or implement based on your auth method
        auth_header = request.headers.get('authorization')
        if auth_header and auth_header.startswith('Bearer '):
            # Extract user from JWT token
            token = auth_header[7:]
            # For development mode, accept any token format
            # In production, this would validate JWT tokens properly
            if self.config.environment.value == "development":
                return f"dev_user_{token[:8]}"  # Simple dev token
            # In production, implement proper JWT validation here
            return None
        return None
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request validation"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = EnterpriseLogger()
        self.security = SecurityHardening()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate request content"""
        client_ip = self._get_client_ip(request)
        
        try:
            # For POST/PUT requests with JSON body
            if request.method in ['POST', 'PUT'] and 'application/json' in request.headers.get('content-type', ''):
                body = await self._get_request_body(request)
                if body:
                    # Validate JSON structure and content
                    is_valid, threats = await self._validate_json_content(body, client_ip, request)
                    if not is_valid:
                        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
                        if critical_threats:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Request contains malicious content"
                            )
            
            # Process request
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(
                "Error in request validation middleware",
                error=e,
                client_ip=client_ip,
                method=request.method,
                path=request.url.path
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request validation failed"
            )
    
    async def _get_request_body(self, request: Request) -> Optional[bytes]:
        """Safely extract request body"""
        try:
            return await request.body()
        except Exception as e:
            self.logger.warning(f"Failed to read request body: {e}")
            return None
    
    async def _validate_json_content(self, body: bytes, client_ip: str, request: Request) -> tuple[bool, list]:
        """Validate JSON content for security threats"""
        try:
            json_data = json.loads(body.decode('utf-8'))
            
            # Extract text content for validation
            text_content = self._extract_text_from_json(json_data)
            user_id = self._get_user_id_from_request(request)
            
            # Validate extracted text
            is_valid, threats = self.security.validate_request(
                query=text_content,
                user_ip=client_ip,
                user_id=user_id
            )
            
            return is_valid, threats
            
        except json.JSONDecodeError:
            return True, []  # Not JSON, skip validation
        except Exception as e:
            self.logger.error(f"JSON validation error: {e}")
            return False, []
    
    def _extract_text_from_json(self, data: Any, max_length: int = 10000) -> str:
        """Recursively extract text content from JSON data"""
        if isinstance(data, str):
            return data[:max_length]
        elif isinstance(data, dict):
            texts = []
            for value in data.values():
                text = self._extract_text_from_json(value, max_length)
                if text:
                    texts.append(text)
                if sum(len(t) for t in texts) > max_length:
                    break
            return ' '.join(texts)[:max_length]
        elif isinstance(data, list):
            texts = []
            for item in data:
                text = self._extract_text_from_json(item, max_length)
                if text:
                    texts.append(text)
                if sum(len(t) for t in texts) > max_length:
                    break
            return ' '.join(texts)[:max_length]
        else:
            return str(data)[:max_length]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else "127.0.0.1"
    
    def _get_user_id_from_request(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Implementation depends on your authentication system
        return None

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and optimization"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = EnterpriseLogger()
        self.request_metrics: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Monitor request performance"""
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Track performance metrics
            metric_key = f"{method}:{path}"
            if metric_key not in self.request_metrics:
                self.request_metrics[metric_key] = []
            
            self.request_metrics[metric_key].append(processing_time)
            
            # Keep only recent metrics (last 100 requests per endpoint)
            if len(self.request_metrics[metric_key]) > 100:
                self.request_metrics[metric_key] = self.request_metrics[metric_key][-100:]
            
            # Log performance metrics
            self.logger.performance(
                f"{method} {path}",
                processing_time,
                status_code=response.status_code,
                method=method,
                path=path
            )
            
            # Alert on slow requests (>5 seconds)
            if processing_time > 5.0:
                self.logger.warning(
                    f"Slow request detected: {method} {path}",
                    processing_time=processing_time,
                    threshold=5.0
                )
            
            # Add performance headers
            response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "Request processing failed",
                error=e,
                method=method,
                path=path,
                processing_time=processing_time
            )
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        metrics = {}
        
        for endpoint, times in self.request_metrics.items():
            if times:
                metrics[endpoint] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'p95_time': sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times)
                }
        
        return metrics

class APIMiddlewareSetup:
    """Central setup for all API middleware"""
    
    @staticmethod
    def setup_middleware(app: FastAPI, config: Optional[Dict[str, Any]] = None):
        """Setup all middleware for the FastAPI application"""
        enterprise_config = get_config()
        
        # CORS middleware - Enhanced for mobile compatibility
        if enterprise_config.security.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"] if enterprise_config.environment.value == "development" else enterprise_config.security.allowed_origins,
                allow_credentials=False if enterprise_config.environment.value == "development" else True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
                allow_headers=[
                    "Accept",
                    "Accept-Language", 
                    "Content-Language",
                    "Content-Type",
                    "Authorization",
                    "X-Requested-With",
                    "Origin",
                    "Access-Control-Request-Method",
                    "Access-Control-Request-Headers",
                    "Cache-Control",
                    "Pragma"
                ],
                expose_headers=["X-Processing-Time", "X-Request-ID"],
                max_age=3600
            )
        
        # GZip compression
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Performance monitoring
        app.add_middleware(PerformanceMiddleware)
        
        # Request validation
        app.add_middleware(RequestValidationMiddleware)
        
        # Security middleware (should be last to process first)
        security_config = SecurityConfig(
            max_request_size=enterprise_config.security.rate_limit_requests_per_minute * 1024 * 1024,
            rate_limit_max_requests=enterprise_config.security.rate_limit_requests_per_minute,
            session_timeout=enterprise_config.security.session_timeout_minutes * 60
        )
        app.add_middleware(SecurityMiddleware, security_config=security_config)
        
        # Global exception handler
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            logger = EnterpriseLogger()
            
            # Log security-relevant HTTP exceptions
            if exc.status_code in [403, 429, 401]:
                client_ip = SecurityMiddleware(app)._get_client_ip(request)
                logger.security(
                    SecurityEventType.AUTHORIZATION,
                    f"HTTP {exc.status_code}: {exc.detail}",
                    severity="medium",
                    source_ip=client_ip,
                    status_code=exc.status_code
                )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger = EnterpriseLogger()
            logger.error(
                "Unhandled exception in API",
                error=exc,
                method=request.method,
                path=request.url.path
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "status_code": 500,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

# Dependency for getting current user (can be used in route handlers)
async def get_current_user(request: Request) -> Optional[str]:
    """Dependency to extract current user from request"""
    auth_header = request.headers.get('authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header[7:]
        # For development, return simple user ID
        # In production, implement proper JWT validation
        return f"user_{token[:8]}" if token else None
    return None

# Dependency for security validation
async def validate_request_security(request: Request) -> bool:
    """Dependency for additional security validation in specific routes"""
    security = SecurityHardening()
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    # This is already handled by middleware, but can be used for additional checks
    return not security.is_ip_blocked(client_ip)