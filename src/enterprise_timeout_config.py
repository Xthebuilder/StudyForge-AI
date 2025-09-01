#!/usr/bin/env python3
"""
Enterprise Timeout Configuration System
Provides centralized timeout management for all AI agents
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import threading
import time


@dataclass
class NetworkTimeouts:
    """Network-related timeout configuration"""
    connection_timeout: int = 10      # TCP connection timeout
    read_timeout: int = 300           # Socket read timeout (5 minutes)
    write_timeout: int = 30           # Socket write timeout
    total_request_timeout: int = 600  # Total request timeout (10 minutes)
    dns_timeout: int = 5              # DNS resolution timeout
    keep_alive_timeout: int = 115     # Keep-alive timeout


@dataclass
class SessionTimeouts:
    """Session management timeout configuration"""
    idle_timeout: int = 1800          # 30 minutes idle before timeout
    max_session_duration: int = 14400 # 4 hours maximum session
    heartbeat_interval: int = 60      # Heartbeat every minute
    session_cleanup_interval: int = 300 # Clean up sessions every 5 minutes
    max_concurrent_sessions: int = 50 # Maximum concurrent sessions


@dataclass
class AIModelTimeouts:
    """AI model-specific timeout configuration"""
    model_load_timeout: int = 180     # 3 minutes to load model
    inference_timeout: int = 300      # 5 minutes for AI inference
    context_processing_timeout: int = 120 # 2 minutes for context processing
    streaming_chunk_timeout: int = 30 # Timeout between streaming chunks
    model_switch_timeout: int = 60    # Time to switch between models


@dataclass
class RetryConfiguration:
    """Retry logic configuration"""
    max_retries: int = 3              # Maximum retry attempts
    base_delay: float = 1.0           # Base delay between retries
    max_delay: float = 60.0           # Maximum delay between retries
    exponential_backoff: bool = True  # Use exponential backoff
    jitter: bool = True               # Add random jitter to delays
    retry_on_timeout: bool = True     # Retry on timeout errors
    retry_on_connection_error: bool = True # Retry on connection errors


@dataclass
class MonitoringConfiguration:
    """Monitoring and alerting configuration"""
    enable_metrics: bool = True       # Enable timeout metrics collection
    alert_on_repeated_timeouts: bool = True # Alert on repeated failures
    timeout_threshold_alerts: int = 5 # Alert after N timeouts
    metrics_export_interval: int = 60 # Export metrics every minute
    log_all_timeouts: bool = True     # Log all timeout events
    performance_monitoring: bool = True # Monitor performance metrics


@dataclass 
class EnterpriseTimeoutConfig:
    """Master timeout configuration for enterprise deployment"""
    network: NetworkTimeouts = field(default_factory=NetworkTimeouts)
    session: SessionTimeouts = field(default_factory=SessionTimeouts)
    ai_model: AIModelTimeouts = field(default_factory=AIModelTimeouts)
    retry: RetryConfiguration = field(default_factory=RetryConfiguration)
    monitoring: MonitoringConfiguration = field(default_factory=MonitoringConfiguration)
    
    # Environment-specific overrides
    environment: str = "production"   # development, staging, production
    debug_timeouts: bool = False      # Enable timeout debugging
    
    def __post_init__(self):
        """Apply environment-specific adjustments"""
        if self.environment == "development":
            # 5-minute timeout for development as requested
            self.network.total_request_timeout = 600   # 10 minutes
            self.ai_model.inference_timeout = 300      # 5 minutes max response time
            self.session.idle_timeout = 3600           # 1 hour
            self.debug_timeouts = True
            
        elif self.environment == "staging":
            # 5-minute timeout for staging as well
            self.network.total_request_timeout = 600   # 10 minutes
            self.ai_model.inference_timeout = 300      # 5 minutes max response time
            self.session.idle_timeout = 2700           # 45 minutes
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnterpriseTimeoutConfig':
        """Create from dictionary"""
        return cls(
            network=NetworkTimeouts(**data.get('network', {})),
            session=SessionTimeouts(**data.get('session', {})),
            ai_model=AIModelTimeouts(**data.get('ai_model', {})),
            retry=RetryConfiguration(**data.get('retry', {})),
            monitoring=MonitoringConfiguration(**data.get('monitoring', {})),
            environment=data.get('environment', 'production'),
            debug_timeouts=data.get('debug_timeouts', False)
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EnterpriseTimeoutConfig':
        """Load configuration from JSON file"""
        if not Path(filepath).exists():
            # Create default config if file doesn't exist
            config = cls()
            config.save_to_file(filepath)
            return config
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            return cls.from_dict(data)


class TimeoutManager:
    """Enterprise-grade timeout manager"""
    
    def __init__(self, config: EnterpriseTimeoutConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TimeoutManager")
        
        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.RLock()
        
        # Metrics
        self.timeout_metrics: Dict[str, int] = {
            'total_timeouts': 0,
            'network_timeouts': 0,
            'ai_timeouts': 0,
            'session_timeouts': 0,
            'retries_exhausted': 0
        }
        
        # Background threads
        self.is_running = True
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Session cleanup thread
        cleanup_thread = threading.Thread(target=self._session_cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        # Metrics export thread if enabled
        if self.config.monitoring.enable_metrics:
            metrics_thread = threading.Thread(target=self._metrics_export_loop, daemon=True)
            metrics_thread.start()
    
    def _session_cleanup_loop(self):
        """Background session cleanup"""
        while self.is_running:
            try:
                self._cleanup_expired_sessions()
                time.sleep(self.config.session.session_cleanup_interval)
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
    
    def _metrics_export_loop(self):
        """Background metrics export"""
        while self.is_running:
            try:
                self._export_metrics()
                time.sleep(self.config.monitoring.metrics_export_interval)
            except Exception as e:
                self.logger.error(f"Metrics export error: {e}")
    
    def create_session(self, session_id: str, user_id: str = "anonymous") -> bool:
        """Create a new session with timeout tracking"""
        with self.session_lock:
            if len(self.sessions) >= self.config.session.max_concurrent_sessions:
                self.logger.warning(f"Maximum concurrent sessions reached: {len(self.sessions)}")
                return False
            
            self.sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'request_count': 0,
                'timeout_count': 0,
                'total_processing_time': 0.0
            }
            
            if self.config.debug_timeouts:
                self.logger.info(f"Created session {session_id} for user {user_id}")
            
            return True
    
    def update_session_activity(self, session_id: str):
        """Update session activity timestamp"""
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_activity'] = datetime.now()
    
    def record_timeout(self, session_id: str, timeout_type: str, details: str = ""):
        """Record a timeout event"""
        with self.session_lock:
            # Update session stats
            if session_id in self.sessions:
                self.sessions[session_id]['timeout_count'] += 1
            
            # Update global metrics
            self.timeout_metrics['total_timeouts'] += 1
            if timeout_type in ['network', 'connection']:
                self.timeout_metrics['network_timeouts'] += 1
            elif timeout_type in ['ai', 'inference', 'model']:
                self.timeout_metrics['ai_timeouts'] += 1
            elif timeout_type == 'session':
                self.timeout_metrics['session_timeouts'] += 1
        
        # Log timeout event
        if self.config.monitoring.log_all_timeouts:
            self.logger.warning(f"Timeout in session {session_id}: {timeout_type} - {details}")
        
        # Check if we need to alert
        if (self.config.monitoring.alert_on_repeated_timeouts and 
            session_id in self.sessions and 
            self.sessions[session_id]['timeout_count'] >= self.config.monitoring.timeout_threshold_alerts):
            self._trigger_timeout_alert(session_id, timeout_type)
    
    def _trigger_timeout_alert(self, session_id: str, timeout_type: str):
        """Trigger timeout alert"""
        self.logger.error(f"ALERT: Repeated timeouts in session {session_id} ({timeout_type})")
        # In production, this would integrate with monitoring systems like Prometheus/Grafana
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        expired_sessions = []
        
        with self.session_lock:
            for session_id, session_data in self.sessions.items():
                # Check idle timeout
                idle_time = (now - session_data['last_activity']).total_seconds()
                if idle_time > self.config.session.idle_timeout:
                    expired_sessions.append((session_id, 'idle'))
                    continue
                
                # Check max session duration
                session_time = (now - session_data['created_at']).total_seconds()
                if session_time > self.config.session.max_session_duration:
                    expired_sessions.append((session_id, 'max_duration'))
            
            # Remove expired sessions
            for session_id, reason in expired_sessions:
                self.sessions.pop(session_id, None)
                self.record_timeout(session_id, 'session', f'Expired: {reason}')
                if self.config.debug_timeouts:
                    self.logger.info(f"Removed expired session {session_id} ({reason})")
    
    def _export_metrics(self):
        """Export timeout metrics"""
        if self.config.debug_timeouts:
            self.logger.info(f"Timeout metrics: {self.timeout_metrics}")
            self.logger.info(f"Active sessions: {len(self.sessions)}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id].copy()
            now = datetime.now()
            
            # Calculate remaining times
            idle_remaining = self.config.session.idle_timeout - (now - session['last_activity']).total_seconds()
            session_remaining = self.config.session.max_session_duration - (now - session['created_at']).total_seconds()
            
            session.update({
                'idle_remaining_seconds': max(0, idle_remaining),
                'session_remaining_seconds': max(0, session_remaining),
                'is_active': idle_remaining > 0 and session_remaining > 0
            })
            
            return session
    
    def shutdown(self):
        """Gracefully shutdown the timeout manager"""
        self.is_running = False
        self.logger.info("Timeout manager shutting down")


# Global timeout configuration instance
_global_config: Optional[EnterpriseTimeoutConfig] = None
_global_manager: Optional[TimeoutManager] = None


def get_timeout_config(config_file: str = "timeout_config.json") -> EnterpriseTimeoutConfig:
    """Get global timeout configuration"""
    global _global_config
    if _global_config is None:
        _global_config = EnterpriseTimeoutConfig.load_from_file(config_file)
    return _global_config


def get_timeout_manager(config_file: str = "timeout_config.json") -> TimeoutManager:
    """Get global timeout manager"""
    global _global_manager
    if _global_manager is None:
        config = get_timeout_config(config_file)
        _global_manager = TimeoutManager(config)
    return _global_manager


def setup_logging_for_timeouts():
    """Setup logging for timeout events"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('timeout_events.log')
        ]
    )


if __name__ == "__main__":
    # Create and save default configuration
    config = EnterpriseTimeoutConfig(environment="development")
    config.save_to_file("timeout_config.json")
    print("✅ Created default enterprise timeout configuration")
    
    # Test timeout manager
    manager = TimeoutManager(config)
    session_id = "test_session_123"
    
    if manager.create_session(session_id, "test_user"):
        print(f"✅ Created test session: {session_id}")
        
        status = manager.get_session_status(session_id)
        print(f"Session status: {status}")
        
        # Simulate a timeout
        manager.record_timeout(session_id, "network", "Connection timed out")
        print("✅ Recorded test timeout event")
    
    print("✅ Enterprise timeout configuration system ready!")