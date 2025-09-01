#!/usr/bin/env python3
"""
StudyForge AI - Web Server
FastAPI-based web interface for StudyForge AI with ChatGPT-style interface
"""

import asyncio
import json
import logging
import socket
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import qrcode
import io
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, status, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Import StudyForge AI components
try:
    from src.web_enhanced_ai import WebEnhancedAI, WebSearchConfig
    from src.enterprise_timeout_config import EnterpriseTimeoutConfig
    from src.database_manager import MemoryConfig
    from src.rag_processor import RAGProcessor, ProcessedDocument, DocumentChunk
    from src.enterprise_logger import EnterpriseLogger
    from src import enterprise_config
    from src.security_hardening import get_security_hardening
    from src.api_middleware import APIMiddlewareSetup, get_current_user, validate_request_security
    from src.health_monitoring import get_health_monitor
except ImportError as e:
    raise ImportError(f"Failed to import StudyForge AI components: {e}")

# Initialize enterprise logging
logger = EnterpriseLogger()

# Configure standard logging as fallback
logging.basicConfig(level=logging.INFO)
std_logger = logging.getLogger(__name__)

# Utility functions
def get_local_ip() -> str:
    """Get the local IP address for LAN access with fallbacks"""
    try:
        # Method 1: Connect to remote address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            logger.debug(f"Local IP detected via remote connection: {local_ip}")
            return local_ip
    except Exception as e:
        logger.debug(f"Remote connection method failed: {e}")
    
    try:
        # Method 2: Use hostname
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip != "127.0.0.1":
            logger.debug(f"Local IP detected via hostname: {local_ip}")
            return local_ip
    except Exception as e:
        logger.debug(f"Hostname method failed: {e}")
    
    logger.warning("Could not detect local IP, using localhost")
    return "127.0.0.1"

# Application lifespan manager with enterprise features
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown with enterprise features"""
    # Startup
    logger.audit('web_server_startup', 'application', outcome='initiated')
    std_logger.info("Starting StudyForge AI Web Server with Enterprise Features...")
    
    try:
        await startup_tasks()
        logger.audit('web_server_startup', 'application', outcome='success')
    except Exception as e:
        logger.critical('Web server startup failed', error=e)
        raise
    
    yield
    
    # Shutdown
    logger.audit('web_server_shutdown', 'application', outcome='initiated')
    std_logger.info("Shutting down StudyForge AI Web Server...")
    
    try:
        await shutdown_tasks()
        logger.audit('web_server_shutdown', 'application', outcome='success')
    except Exception as e:
        logger.error('Web server shutdown error', error=e)

app = FastAPI(
    title="StudyForge AI Enterprise Web Interface", 
    version="1.0.0",
    description="Enterprise-grade AI study companion with comprehensive security, monitoring, and management features",
    lifespan=lifespan
)

# Setup enterprise middleware with comprehensive security
APIMiddlewareSetup.setup_middleware(app)

std_logger.info("Enterprise middleware configured successfully")
logger.audit('middleware_setup', 'application', outcome='success')

# Setup templates and static files with error handling
try:
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.error(f"Failed to setup templates/static files: {e}")
    raise

# Pydantic models with validation
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Chat message content")
    force_web_search: bool = Field(default=False, description="Force web search for this message")
    session_id: Optional[str] = Field(default=None, description="Session ID for chat continuity")
    
    @field_validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        if v is not None:
            # Accept UUID format or generate one from simple string
            if not v:
                return None
            try:
                # Try parsing as UUID first
                uuid.UUID(v)
            except ValueError:
                # If not UUID, check if it's a valid string and convert
                if isinstance(v, str) and len(v.strip()) > 0:
                    # For simple strings like "test123", generate a consistent UUID
                    import hashlib
                    hash_obj = hashlib.md5(v.encode())
                    hex_dig = hash_obj.hexdigest()
                    # Format as UUID
                    v = f"{hex_dig[:8]}-{hex_dig[8:12]}-{hex_dig[12:16]}-{hex_dig[16:20]}-{hex_dig[20:32]}"
                else:
                    raise ValueError('Invalid session ID format')
        return v


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response content")
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Response timestamp")
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    used_web_search: bool = Field(default=False, description="Whether web search was used")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")
    chunks_count: int = Field(..., description="Number of content chunks created")
    keywords: List[str] = Field(..., description="Extracted keywords")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

class DocumentLibraryResponse(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of all documents in library")
    total_count: int = Field(..., description="Total number of documents")

class DeleteDocumentRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to delete")

class DeleteDocumentResponse(BaseModel):
    success: bool = Field(..., description="Whether deletion was successful")
    document_id: str = Field(..., description="ID of deleted document")
    message: str = Field(..., description="Result message")


class DatabaseManager:
    """Enhanced database manager with connection pooling and error handling"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        return conn
    
    def _init_database(self) -> None:
        """Initialize SQLite database with proper indexes"""
        try:
            with self._get_connection() as conn:
                # Create tables
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT DEFAULT 'New Chat'
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        response_time REAL DEFAULT 0,
                        used_web_search BOOLEAN DEFAULT 0,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    )
                ''')
                
                # Create indexes for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active)')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise


class SessionManager(DatabaseManager):
    """Enhanced session manager with better error handling and cleanup"""
    
    def __init__(self, db_path: str = "sessions.db"):
        super().__init__(db_path)
        self.active_sessions: Dict[str, datetime] = {}
        self.cleanup_interval = timedelta(hours=24)  # Cleanup old sessions after 24h
        
    async def cleanup_old_sessions(self) -> None:
        """Remove old inactive sessions"""
        try:
            cutoff_time = datetime.now() - self.cleanup_interval
            with self._get_connection() as conn:
                cursor = conn.execute(
                    'DELETE FROM sessions WHERE last_active < ?',
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old sessions")
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
    
    def create_session(self) -> str:
        """Create a new chat session with proper error handling"""
        session_id = str(uuid.uuid4())
        
        try:
            with self._get_connection() as conn:
                conn.execute(
                    'INSERT INTO sessions (session_id) VALUES (?)',
                    (session_id,)
                )
                conn.commit()
            
            self.active_sessions[session_id] = datetime.now()
            logger.info(f"Created new session: {session_id}")
            return session_id
        
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get message history for a session with proper error handling"""
        if not self._validate_session_id(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID format"
            )
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT role, content, timestamp, response_time, used_web_search
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                ''', (session_id,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'response_time': float(row['response_time']) if row['response_time'] else 0.0,
                        'used_web_search': bool(row['used_web_search'])
                    })
                
                return messages
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting session history: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve session history"
            )
        except Exception as e:
            logger.error(f"Unexpected error getting session history: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def _validate_session_id(self, session_id: str) -> bool:
        """Validate session ID format"""
        try:
            uuid.UUID(session_id)
            return True
        except (ValueError, TypeError):
            return False
    
    def save_message(self, session_id: str, role: str, content: str, 
                    response_time: float = 0, used_web_search: bool = False) -> None:
        """Save a message to the session with proper validation and error handling"""
        if not self._validate_session_id(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID format"
            )
        
        if role not in ['user', 'assistant']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role. Must be 'user' or 'assistant'"
            )
        
        if not content or not content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message content cannot be empty"
            )
        
        try:
            with self._get_connection() as conn:
                # Insert message
                conn.execute('''
                    INSERT INTO messages (session_id, role, content, response_time, used_web_search)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, role, content.strip(), response_time, used_web_search))
                
                # Update session last_active
                conn.execute('''
                    UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?
                ''', (session_id,))
                
                conn.commit()
                
                # Update active sessions tracker
                self.active_sessions[session_id] = datetime.now()
        
        except sqlite3.Error as e:
            logger.error(f"Database error saving message: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save message"
            )
        except Exception as e:
            logger.error(f"Unexpected error saving message: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions for sidebar with proper error handling"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT s.session_id, s.created_at, s.last_active, s.title,
                           (SELECT content FROM messages WHERE session_id = s.session_id 
                            AND role = 'user' ORDER BY timestamp LIMIT 1) as first_message
                    FROM sessions s
                    ORDER BY s.last_active DESC
                    LIMIT 50
                ''')
                
                sessions = []
                for row in cursor.fetchall():
                    first_msg = row['first_message'] or ''
                    title = row['title'] or (
                        first_msg[:50] + '...' if len(first_msg) > 50 else first_msg
                    ) or 'New Chat'
                    
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': row['created_at'],
                        'last_active': row['last_active'],
                        'title': title,
                        'preview': first_msg[:100] + '...' if len(first_msg) > 100 else first_msg or 'No messages'
                    })
                
                return sessions
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting all sessions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve sessions"
            )
        except Exception as e:
            logger.error(f"Unexpected error getting all sessions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session with proper validation"""
        if not self._validate_session_id(session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID format"
            )
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
                if cursor.rowcount == 0:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Session not found"
                    )
                conn.commit()
            
            # Remove from active sessions
            self.active_sessions.pop(session_id, None)
            logger.info(f"Deleted session: {session_id}")
        
        except sqlite3.Error as e:
            logger.error(f"Database error deleting session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete session"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        if not self._validate_session_id(session_id):
            return False
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    'SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1',
                    (session_id,)
                )
                return cursor.fetchone() is not None
        
        except Exception as e:
            logger.error(f"Error checking if session exists: {e}")
            return False


class ConfigManager:
    """Enhanced configuration manager with validation and error handling"""
    
    def __init__(self, config_file: str = "web_config.json"):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file with proper error handling"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Validate and merge with defaults
                    self.config = {**self.get_default_config(), **loaded_config}
                    logger.info("Configuration loaded successfully")
            else:
                self.config = self.get_default_config()
                self.save_config()
                logger.info("Default configuration created")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            self.config = self.get_default_config()
            self.save_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with proper types"""
        return {
            "theme": "dark",
            "auto_search": True,
            "search_threshold": 0.7,
            "max_search_results": 10,
            "timeout_seconds": 30,
            "retry_count": 3,
            "model_name": "gpt-oss:20b",
            "ollama_url": "http://localhost:11434/api/generate"
        }
    
    def save_config(self) -> None:
        """Save configuration to file with error handling"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save configuration"
            )
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with validation"""
        # Validate updates against schema
        valid_keys = set(self.get_default_config().keys())
        for key in updates:
            if key not in valid_keys:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid configuration key: {key}"
                )
        
        # Type validation
        self._validate_config_values(updates)
        
        # Update config
        self.config.update(updates)
        self.save_config()
    
    def _validate_config_values(self, config: Dict[str, Any]) -> None:
        """Validate configuration values"""
        if 'search_threshold' in config:
            threshold = config['search_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="search_threshold must be a number between 0 and 1"
                )
        
        if 'max_search_results' in config:
            max_results = config['max_search_results']
            if not isinstance(max_results, int) or not 1 <= max_results <= 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="max_search_results must be an integer between 1 and 50"
                )
        
        if 'timeout_seconds' in config:
            timeout = config['timeout_seconds']
            if not isinstance(timeout, int) or not 5 <= timeout <= 300:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="timeout_seconds must be an integer between 5 and 300"
                )
        
        if 'retry_count' in config:
            retry = config['retry_count']
            if not isinstance(retry, int) or not 0 <= retry <= 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="retry_count must be an integer between 0 and 10"
                )
        
        if 'theme' in config:
            theme = config['theme']
            if theme not in ['dark', 'light']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="theme must be either 'dark' or 'light'"
                )


# Enterprise Application managers with comprehensive lifecycle management
class EnterpriseAppManagers:
    """Enterprise application managers with full lifecycle management, monitoring, and security"""
    
    def __init__(self):
        self.session_manager: Optional[SessionManager] = None
        self.config_manager: Optional[ConfigManager] = None
        self.rag_processor: Optional[RAGProcessor] = None
        self.ai_instances: Dict[str, WebEnhancedAI] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Enterprise components
        self.enterprise_config = None
        self.security_hardening = None
        self.health_monitor = None
        
        self.logger = EnterpriseLogger()
    
    async def initialize(self):
        """Initialize all managers with enterprise features"""
        try:
            self.logger.audit('managers_initialization', 'application', outcome='initiated')
            
            # Initialize enterprise configuration
            self.enterprise_config = enterprise_config.get_config_manager()
            config = enterprise_config.get_config()
            self.logger.info('Enterprise configuration loaded', environment=config.environment.value)
            
            # Initialize security hardening
            self.security_hardening = get_security_hardening()
            self.logger.info('Security hardening initialized')
            
            # Initialize health monitoring
            self.health_monitor = get_health_monitor()
            await self.health_monitor.start_monitoring(interval_seconds=60)
            self.logger.info('Health monitoring started')
            
            # Initialize application managers
            self.session_manager = SessionManager()
            self.config_manager = ConfigManager()
            self.rag_processor = RAGProcessor()
            
            self.logger.audit('managers_initialization', 'application', outcome='success')
            std_logger.info("Enterprise application managers initialized")
            
        except Exception as e:
            self.logger.critical('Failed to initialize enterprise managers', error=e)
            raise
    
    async def cleanup(self):
        """Cleanup resources with enterprise monitoring"""
        try:
            self.logger.audit('managers_cleanup', 'application', outcome='initiated')
            
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
                self.logger.info('Health monitoring stopped')
            
            # Close WebSocket connections
            for session_id, ws in self.websocket_connections.items():
                try:
                    await ws.close()
                    self.logger.audit('websocket_closed', 'cleanup', session_id=session_id)
                except Exception as e:
                    self.logger.warning(f'Error closing WebSocket for {session_id}', error=e)
            
            # Cleanup AI instances
            for session_id, ai in self.ai_instances.items():
                if hasattr(ai, 'cleanup'):
                    try:
                        await ai.cleanup()
                        self.logger.audit('ai_instance_cleaned', 'cleanup', session_id=session_id)
                    except Exception as e:
                        self.logger.warning(f'Error cleaning AI instance for {session_id}', error=e)
            
            # Cleanup old sessions
            if self.session_manager:
                await self.session_manager.cleanup_old_sessions()
            
            self.logger.audit('managers_cleanup', 'application', outcome='success')
            std_logger.info("Enterprise application cleanup completed")
            
        except Exception as e:
            self.logger.error('Error during enterprise cleanup', error=e)

# Global enterprise managers instance
managers = EnterpriseAppManagers()


def get_local_ip() -> str:
    """Get the local IP address for LAN access with fallback"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            logger.info(f"Detected local IP: {local_ip}")
            return local_ip
    except Exception as e:
        logger.warning(f"Failed to detect local IP: {e}, using localhost")
        return "127.0.0.1"


def generate_qr_code(url: str) -> str:
    """Generate QR code for URL as base64 string with error handling"""
    try:
        if not url or not url.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL format")
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        qr_base64 = base64.b64encode(buffer.getvalue()).decode()
        logger.info(f"Generated QR code for URL: {url}")
        return qr_base64
    
    except Exception as e:
        logger.error(f"Failed to generate QR code: {e}")
        return ""


async def get_ai_instance(session_id: str) -> WebEnhancedAI:
    """Get or create AI instance for session with proper error handling"""
    try:
        if session_id not in managers.ai_instances:
            logger.info(f"Creating new AI instance for session: {session_id}")
            ai = WebEnhancedAI(user_id=session_id, rag_processor=managers.rag_processor)
            
            # Initialize with timeout
            init_success = await asyncio.wait_for(
                ai.initialize(), 
                timeout=30.0
            )
            
            if not init_success:
                raise RuntimeError("Failed to initialize AI instance")
            
            managers.ai_instances[session_id] = ai
            logger.info(f"AI instance created successfully for session: {session_id}")
        
        return managers.ai_instances[session_id]
    
    except asyncio.TimeoutError:
        logger.error(f"Timeout initializing AI instance for session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI initialization timeout"
        )
    except Exception as e:
        logger.error(f"Failed to get AI instance for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize AI instance"
        )


# Enterprise application lifecycle functions
async def startup_tasks():
    """Enterprise startup tasks with comprehensive initialization"""
    try:
        await managers.initialize()
        
        # Log startup completion with system info
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        logger.info('StudyForge AI Enterprise Web Server started successfully', **system_info)
        std_logger.info("StudyForge AI Enterprise Web Server started successfully")
        
    except Exception as e:
        logger.critical('Enterprise startup failed', error=e)
        std_logger.error(f"Enterprise startup failed: {e}")
        raise

async def shutdown_tasks():
    """Enterprise shutdown tasks with proper cleanup"""
    try:
        await managers.cleanup()
        logger.info('StudyForge AI Enterprise Web Server shutdown completed')
        std_logger.info("StudyForge AI Enterprise Web Server shutdown completed")
        
    except Exception as e:
        logger.error('Enterprise shutdown error', error=e)
        std_logger.error(f"Enterprise shutdown error: {e}")

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            detail=exc.detail
        ).dict()
    )

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface with error handling"""
    try:
        local_ip = get_local_ip()
        port = 8000  # This should be configurable
        url = f"http://{local_ip}:{port}"
        qr_code = generate_qr_code(url)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "local_ip": local_ip,
            "port": port,
            "qr_code": qr_code,
            "config": managers.config_manager.config if managers.config_manager else {}
        })
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load main interface"
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, request: Request):
    """Process chat message with enterprise security and monitoring"""
    if not managers.session_manager:
        logger.error('Chat endpoint called but session manager not initialized')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    start_time = time.time()
    session_id = None
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    # Enterprise security validation
    if managers.security_hardening:
        is_valid, threats = managers.security_hardening.validate_request(
            query=message.message,
            user_ip=client_ip,
            user_id=message.session_id
        )
        
        if not is_valid:
            logger.security(
                event_type='suspicious_activity',
                message='Malicious chat request blocked',
                severity='high',
                source_ip=client_ip,
                threats_count=len(threats)
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request blocked due to security policy"
            )
    
    request_id = f"chat_{int(time.time() * 1000)}"
    with logger.request_context(request_id, user_id=message.session_id, ip_address=client_ip):
        logger.audit('chat_request_received', 'chat_api', 
                    message_length=len(message.message),
                    force_web_search=message.force_web_search)
        
        try:
            # Create or validate session
            if not message.session_id:
                session_id = managers.session_manager.create_session()
                logger.info(f"Created new session: {session_id}")
            else:
                session_id = message.session_id
                if not managers.session_manager.session_exists(session_id):
                    logger.warning(f'Session not found: {session_id}')
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Session not found"
                    )
        
            # Get AI instance for this session
            ai = await get_ai_instance(session_id)
            
            # Save user message
            managers.session_manager.save_message(session_id, "user", message.message)
            logger.audit('user_message_saved', 'session_manager', session_id=session_id)
            
            # Get AI response with enterprise timeout configuration
            ent_config = enterprise_config.get_config()
            timeout_seconds = ent_config.ai_model.timeout_seconds
            
            try:
                response = await asyncio.wait_for(
                    ai.query_with_web_enhancement(
                        message.message, 
                        force_web_search=message.force_web_search
                    ),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"AI query timeout for session: {session_id}", timeout=timeout_seconds)
                logger.security(
                    event_type='suspicious_activity',
                    message='AI query timeout - potential resource exhaustion',
                    severity='medium',
                    source_ip=client_ip,
                    session_id=session_id
                )
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="AI response timeout"
                )
        
            response_time = time.time() - start_time
            
            # Determine if web search was used
            used_web_search = (
                message.force_web_search or 
                "search results:" in response.lower() or
                "web search" in response.lower()
            )
            
            # Save AI response
            managers.session_manager.save_message(
                session_id, "assistant", response, 
                response_time=response_time,
                used_web_search=used_web_search
            )
            
            # Log successful chat completion
            logger.audit('chat_response_completed', 'chat_api',
                        outcome='success',
                        session_id=session_id,
                        response_time=response_time,
                        response_length=len(response),
                        used_web_search=used_web_search)
            
            logger.performance('chat_request', response_time,
                             session_id=session_id,
                             message_length=len(message.message),
                             response_length=len(response))
            
            std_logger.info(f"Chat response generated for session {session_id} in {response_time:.2f}s")
            
            return ChatResponse(
                response=response,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                used_web_search=used_web_search,
                retry_count=getattr(ai, 'last_retry_count', 0)
            )
        
        except HTTPException:
            raise
        except Exception as e:
            error_time = time.time() - start_time
            
            logger.error('Unexpected error in chat endpoint', error=e, 
                        session_id=session_id,
                        processing_time=error_time)
            
            # Security monitoring for unusual errors
            if any(term in str(e).lower() for term in ['injection', 'exploit', 'attack', 'malicious']):
                logger.security(
                    event_type='suspicious_activity',
                    message=f'Potential security issue in chat: {str(e)[:100]}',
                    severity='high',
                    source_ip=client_ip,
                    session_id=session_id
                )
            
            # Try to save error message if we have a session
            if session_id and managers.session_manager:
                try:
                    managers.session_manager.save_message(
                        session_id, "assistant", 
                        f"Sorry, I encountered an error: {str(e)[:100]}...",
                        response_time=error_time,
                        used_web_search=False
                    )
                    logger.audit('error_message_saved', 'session_manager', session_id=session_id)
                except Exception:
                    pass  # Don't let error saving cause another error
            
            return ChatResponse(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                session_id=session_id or "error",
                timestamp=datetime.now().isoformat(),
                response_time=error_time,
                used_web_search=False,
                retry_count=0,
                error=str(e)[:200]  # Truncate error message
            )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming responses with enhanced error handling"""
    if not managers.session_manager:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Service not initialized")
        return
    
    # Validate session ID format
    if not managers.session_manager._validate_session_id(session_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid session ID")
        return
    
    await websocket.accept()
    managers.websocket_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        # Ensure session exists or create it
        if not managers.session_manager.session_exists(session_id):
            logger.info(f"Creating session for WebSocket: {session_id}")
            # Note: This might fail if session_id is not a valid UUID generated by us
            # In a real application, you might want to create a new session instead
        
        ai = await get_ai_instance(session_id)
        
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)  # 5 min timeout
                logger.debug(f"WebSocket received data: {data}")
                
                message = data.get("message", "").strip()
                force_web_search = data.get("force_web_search", False)
                
                if not message:
                    logger.warning(f"Empty message received from WebSocket. Raw data: {data}")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Empty message received"
                    })
                    continue
                
                if len(message) > 4000:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Message too long (max 4000 characters)"
                    })
                    continue
                
                # Save user message
                managers.session_manager.save_message(session_id, "user", message)
                
                # Send typing indicator
                await websocket.send_json({"type": "typing", "status": True})
                
                start_time = time.time()
                
                try:
                    # Use streaming response for better UX
                    response = await ai.query_with_streaming(message, websocket, force_web_search)
                    response_time = time.time() - start_time
                    
                    # Determine if web search was used
                    used_web_search = (
                        force_web_search or 
                        "search results:" in response.lower() or
                        "web search" in response.lower()
                    )
                    
                    # Send final response confirmation
                    await websocket.send_json({
                        "type": "final",
                        "response": response,
                        "response_time": response_time,
                        "used_web_search": used_web_search
                    })
                    
                    # Save AI response
                    managers.session_manager.save_message(
                        session_id, "assistant", response,
                        response_time=response_time,
                        used_web_search=used_web_search
                    )
                    
                    logger.info(f"WebSocket response sent for session {session_id} in {response_time:.2f}s")
                    
                except asyncio.TimeoutError:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Response timeout - please try again"
                    })
                    logger.warning(f"WebSocket response timeout for session: {session_id}")
                    
                except Exception as e:
                    error_msg = f"AI processing error: {str(e)[:100]}..."
                    await websocket.send_json({
                        "type": "error",
                        "error": error_msg
                    })
                    logger.error(f"WebSocket AI error for session {session_id}: {e}")
                
                finally:
                    # Stop typing indicator
                    try:
                        await websocket.send_json({"type": "typing", "status": False})
                    except Exception:
                        pass  # Connection might be closed
                        
            except asyncio.TimeoutError:
                logger.info(f"WebSocket idle timeout for session: {session_id}")
                await websocket.send_json({
                    "type": "info",
                    "message": "Connection idle timeout"
                })
                break
                
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Failed to process message"
                })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket unexpected error: {e}")
    finally:
        # Cleanup
        managers.websocket_connections.pop(session_id, None)
        logger.info(f"WebSocket cleanup completed for session: {session_id}")


@app.get("/api/sessions")
async def get_sessions():
    """Get all chat sessions with error handling"""
    if not managers.session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized"
        )
    
    try:
        sessions = managers.session_manager.get_all_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get history for a specific session with validation"""
    if not managers.session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized"
        )
    
    try:
        history = managers.session_manager.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    if not managers.config_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration manager not initialized"
        )
    
    return managers.config_manager.config


@app.post("/api/config")
async def update_config(updates: dict):
    """Update configuration with validation"""
    if not managers.config_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration manager not initialized"
        )
    
    try:
        managers.config_manager.update_config(updates)
        logger.info(f"Configuration updated: {list(updates.keys())}")
        return {"status": "updated", "config": managers.config_manager.config}
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise


@app.get("/api/analytics/{session_id}")
async def get_analytics(session_id: str):
    """Get analytics for a session with proper validation"""
    if not managers.session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized"
        )
    
    # Validate session ID
    if not managers.session_manager._validate_session_id(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )
    
    try:
        # Check if session exists
        if not managers.session_manager.session_exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Get AI instance stats if available
        ai = managers.ai_instances.get(session_id)
        if ai:
            stats = getattr(ai, 'stats', {})
            return {
                "session_id": session_id,
                "total_queries": stats.get('total_queries', 0),
                "web_searches": stats.get('web_searches', 0),
                "average_response_time": stats.get('avg_response_time', 0),
                "total_response_time": stats.get('total_response_time', 0),
                "errors": stats.get('errors', 0),
                "session_active": True
            }
        else:
            # Return basic analytics for inactive sessions
            return {
                "session_id": session_id,
                "total_queries": 0,
                "web_searches": 0,
                "average_response_time": 0,
                "total_response_time": 0,
                "errors": 0,
                "session_active": False
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session analytics"
        )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session with proper validation and cleanup"""
    if not managers.session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized"
        )
    
    try:
        # Delete session using session manager (includes validation)
        managers.session_manager.delete_session(session_id)
        
        # Cleanup AI instance if exists
        if session_id in managers.ai_instances:
            ai = managers.ai_instances.pop(session_id)
            if hasattr(ai, 'cleanup'):
                try:
                    await ai.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up AI instance: {e}")
        
        # Close WebSocket connection if exists
        if session_id in managers.websocket_connections:
            ws = managers.websocket_connections.pop(session_id)
            try:
                await ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        logger.info(f"Session deleted successfully: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


# RAG Document Management Endpoints
@app.post("/api/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Upload and process a document for RAG with enterprise security"""
    client_ip = request.client.host if request.client else "127.0.0.1"
    upload_id = f"upload_{int(time.time() * 1000)}"
    
    with logger.request_context(upload_id, user_id=session_id, ip_address=client_ip):
        logger.audit('document_upload_initiated', 'upload_api',
                    filename=file.filename,
                    content_type=file.content_type,
                    session_id=session_id)
        
        try:
            # Enterprise security validation
            if managers.security_hardening:
                # Validate filename
                is_valid, threats = managers.security_hardening.validate_request(
                    filename=file.filename,
                    user_ip=client_ip,
                    user_id=session_id
                )
                
                if not is_valid:
                    logger.security(
                        event_type='malicious_file',
                        message='Malicious file upload blocked',
                        severity='high',
                        source_ip=client_ip,
                        filename=file.filename
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File blocked due to security policy"
                    )
            
            # Validate file size with enterprise configuration
            ent_config = enterprise_config.get_config()
            max_file_size = getattr(ent_config.monitoring.alert_thresholds, 'max_file_size', 50 * 1024 * 1024)
            
            file_content = await file.read()
            
            if len(file_content) > max_file_size:
                logger.warning('File size limit exceeded', 
                             file_size=len(file_content),
                             max_size=max_file_size,
                             filename=file.filename)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds {max_file_size // (1024*1024)}MB limit"
                )
            
            # Additional content validation
            if managers.security_hardening:
                content_valid, content_threats = managers.security_hardening.validate_request(
                    file_content=file_content,
                    filename=file.filename,
                    user_ip=client_ip,
                    user_id=session_id
                )
                
                if not content_valid:
                    logger.security(
                        event_type='malicious_file',
                        message='Malicious file content detected',
                        severity='critical',
                        source_ip=client_ip,
                        filename=file.filename
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File content blocked due to security policy"
                    )
        
            # Detect content type
            import mimetypes
            content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'text/plain'
            
            # Process document
            start_process_time = time.time()
            doc = await managers.rag_processor.process_and_store_document(
                file_content, file.filename, content_type
            )
            processing_time = time.time() - start_process_time
            
            # Log successful upload
            logger.audit('document_upload_completed', 'upload_api',
                        outcome='success',
                        document_id=doc.document_id,
                        filename=doc.filename,
                        file_size=doc.file_size,
                        processing_time=processing_time,
                        chunks_count=doc.chunks_count)
            
            logger.performance('document_upload', processing_time,
                             file_size=doc.file_size,
                             chunks_count=doc.chunks_count,
                             filename=file.filename)
            
            std_logger.info(f"Document uploaded and processed: {doc.document_id}")
            
            return DocumentUploadResponse(
                document_id=doc.document_id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                chunks_count=doc.chunks_count,
                keywords=doc.keywords,
                processing_time=doc.processing_time,
                success=doc.error_message is None,
                error_message=doc.error_message
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error('Document upload error', error=e, 
                        filename=file.filename,
                        file_size=len(file_content) if 'file_content' in locals() else 0)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process document"
            )

@app.get("/api/document-library", response_model=DocumentLibraryResponse)
async def get_document_library():
    """Get all documents in the RAG library"""
    try:
        documents = managers.rag_processor.get_document_library()
        
        doc_list = []
        for doc in documents:
            doc_dict = {
                'document_id': doc.document_id,
                'filename': doc.filename,
                'file_type': doc.file_type,
                'file_size': doc.file_size,
                'content_preview': doc.content_preview,
                'chunks_count': doc.chunks_count,
                'keywords': doc.keywords,
                'processing_time': doc.processing_time,
                'upload_time': doc.upload_time,
                'error_message': doc.error_message
            }
            doc_list.append(doc_dict)
        
        return DocumentLibraryResponse(
            documents=doc_list,
            total_count=len(doc_list)
        )
        
    except Exception as e:
        logger.error(f"Error fetching document library: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch document library"
        )

@app.delete("/api/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    """Delete a document from the RAG library"""
    try:
        success = managers.rag_processor.delete_document(document_id)
        
        if success:
            logger.info(f"Document deleted: {document_id}")
            return DeleteDocumentResponse(
                success=True,
                document_id=document_id,
                message="Document deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if managers are initialized
        if not managers.session_manager:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Service not initialized",
                    "timestamp": datetime.now().isoformat(),
                    "components": []
                }
            )
        
        components = []
        overall_status = "healthy"
        
        # Check database connectivity
        try:
            managers.session_manager.get_all_sessions()
            components.append({
                "name": "database",
                "status": "healthy",
                "message": "Database connection working"
            })
        except Exception as e:
            components.append({
                "name": "database", 
                "status": "unhealthy",
                "message": f"Database error: {str(e)[:100]}"
            })
            overall_status = "degraded"
        
        # Check AI model availability (simple check)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        components.append({
                            "name": "ai_model",
                            "status": "healthy", 
                            "message": "AI model service available"
                        })
                    else:
                        components.append({
                            "name": "ai_model",
                            "status": "unhealthy",
                            "message": f"AI service returned {response.status}"
                        })
                        overall_status = "degraded"
        except Exception as e:
            components.append({
                "name": "ai_model",
                "status": "unhealthy", 
                "message": f"AI service error: {str(e)[:100]}"
            })
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)[:100]}",
                "timestamp": datetime.now().isoformat(),
                "components": []
            }
        )

@app.get("/api/supported-file-types")
async def get_supported_file_types():
    """Get list of supported file types for RAG"""
    try:
        file_types = managers.rag_processor.get_supported_file_types()
        
        # Add human-readable descriptions
        type_descriptions = {
            'text/plain': 'Plain Text (.txt)',
            'application/pdf': 'PDF Documents (.pdf)',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word Documents (.docx)',
            'text/html': 'HTML Files (.html)',
            'text/markdown': 'Markdown Files (.md)',
            'application/json': 'JSON Files (.json)',
            'text/csv': 'CSV Files (.csv)',
            'text/xml': 'XML Files (.xml)'
        }
        
        supported_types = []
        for file_type in file_types:
            supported_types.append({
                'mime_type': file_type,
                'description': type_descriptions.get(file_type, file_type)
            })
        
        return {
            'supported_types': supported_types,
            'max_file_size_mb': 50
        }
        
    except Exception as e:
        logger.error(f"Error fetching supported file types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch supported file types"
        )


# Note: @app.on_event is deprecated in favor of lifespan context manager
# The startup and shutdown logic is now handled in the lifespan function at the top


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get local IP for binding
    local_ip = get_local_ip()
    port = 8000
    
    logger.info("🚀 Starting StudyForge AI Web Server...")
    logger.info(f"💻 Local access: http://localhost:{port}")
    logger.info(f"🌐 LAN access: http://{local_ip}:{port}")
    logger.info(f"📱 Mobile: Scan QR code at http://{local_ip}:{port}")
    
    # Run the server with proper configuration and enhanced signal handling
    try:
        import signal
        import sys
        
        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully"""
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.audit('web_server_shutdown_requested', 'signal_handler', signal=signal_name)
            std_logger.info(f"Shutdown signal received: {signal_name}")
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        uvicorn.run(
            "web_server:app",
            host="0.0.0.0",  # Bind to all interfaces for LAN access
            port=port,
            reload=False,
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.audit('web_server_stopped', 'main', reason='user_interrupt')
        std_logger.info("Server stopped by user (Ctrl+C)")
    except SystemExit:
        logger.audit('web_server_stopped', 'main', reason='system_exit')
        std_logger.info("Server stopped by system")
    except Exception as e:
        logger.critical('Web server startup failed', error=e)
        std_logger.error(f"Server error: {e}")
        raise