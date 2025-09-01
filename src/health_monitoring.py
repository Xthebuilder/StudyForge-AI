"""
Health Check and Monitoring System for StudyForge AI
Provides comprehensive system health monitoring, metrics collection,
and alerting capabilities for enterprise deployments
"""

import asyncio
import time
import psutil
import platform
import sqlite3
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from pathlib import Path

from .enterprise_logger import EnterpriseLogger
from .enterprise_config import get_config

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Types of system components to monitor"""
    DATABASE = "database"
    AI_MODEL = "ai_model"
    WEB_SEARCH = "web_search"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    component_type: ComponentType
    check_function: Callable
    timeout_seconds: int = 30
    critical: bool = False
    enabled: bool = True
    interval_seconds: int = 60
    
@dataclass
class HealthResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    uptime_seconds: float
    
class DatabaseHealthChecker:
    """Database connectivity and performance health checks"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
    
    async def check_sqlite_health(self) -> HealthResult:
        """Check SQLite database health"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            conn = sqlite3.connect('sessions.db', timeout=5)
            cursor = conn.cursor()
            
            # Test query performance
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            result = cursor.fetchone()
            
            # Test write performance
            cursor.execute("CREATE TEMP TABLE health_check (id INTEGER)")
            cursor.execute("INSERT INTO health_check (id) VALUES (1)")
            cursor.execute("DROP TABLE health_check")
            
            conn.commit()
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time > 1000:  # > 1 second
                return HealthResult(
                    name="sqlite_database",
                    status=HealthStatus.DEGRADED,
                    message=f"Database responding slowly: {response_time:.2f}ms",
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    details={"query_time_ms": response_time}
                )
            
            return HealthResult(
                name="sqlite_database",
                status=HealthStatus.HEALTHY,
                message="Database operational",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={"tables": result[0] if result else 0}
            )
            
        except sqlite3.OperationalError as e:
            return HealthResult(
                name="sqlite_database",
                status=HealthStatus.CRITICAL,
                message="Database connection failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
        except Exception as e:
            return HealthResult(
                name="sqlite_database",
                status=HealthStatus.UNHEALTHY,
                message="Database health check failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

class AIModelHealthChecker:
    """AI model service health checks"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
        self.config = get_config()
    
    async def check_ollama_health(self) -> HealthResult:
        """Check Ollama AI model service health"""
        start_time = time.time()
        base_url = self.config.ai_model.base_url
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Check if Ollama is running
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        
                        # Check if our specific model is available
                        model_available = any(
                            model.get('name') == self.config.ai_model.model_name 
                            for model in models
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        if not model_available:
                            return HealthResult(
                                name="ollama_ai_model",
                                status=HealthStatus.DEGRADED,
                                message=f"Model {self.config.ai_model.model_name} not available",
                                timestamp=datetime.now(),
                                response_time_ms=response_time,
                                details={
                                    "available_models": [m.get('name') for m in models],
                                    "requested_model": self.config.ai_model.model_name
                                }
                            )
                        
                        # Test model inference with simple query
                        test_result = await self._test_model_inference(session)
                        if not test_result:
                            return HealthResult(
                                name="ollama_ai_model",
                                status=HealthStatus.DEGRADED,
                                message="Model inference test failed",
                                timestamp=datetime.now(),
                                response_time_ms=response_time,
                                details={"model_count": len(models)}
                            )
                        
                        return HealthResult(
                            name="ollama_ai_model",
                            status=HealthStatus.HEALTHY,
                            message="AI model service operational",
                            timestamp=datetime.now(),
                            response_time_ms=response_time,
                            details={
                                "model_count": len(models),
                                "active_model": self.config.ai_model.model_name
                            }
                        )
                    else:
                        return HealthResult(
                            name="ollama_ai_model",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Ollama API returned status {response.status}",
                            timestamp=datetime.now(),
                            response_time_ms=(time.time() - start_time) * 1000,
                            error=f"HTTP {response.status}"
                        )
                        
        except aiohttp.ClientError as e:
            return HealthResult(
                name="ollama_ai_model",
                status=HealthStatus.CRITICAL,
                message="Cannot connect to AI model service",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
        except Exception as e:
            return HealthResult(
                name="ollama_ai_model",
                status=HealthStatus.UNHEALTHY,
                message="AI model health check failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _test_model_inference(self, session: aiohttp.ClientSession) -> bool:
        """Test model inference with a simple query"""
        try:
            test_data = {
                "model": self.config.ai_model.model_name,
                "prompt": "Test",
                "stream": False,
                "options": {"max_tokens": 10}
            }
            
            async with session.post(
                f"{self.config.ai_model.base_url}/api/generate",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return bool(result.get('response'))
                return False
                
        except Exception as e:
            self.logger.warning(f"Model inference test failed: {e}")
            return False

class SystemResourceChecker:
    """System resource monitoring (CPU, Memory, Disk)"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
        self.config = get_config()
    
    async def check_cpu_health(self) -> HealthResult:
        """Check CPU utilization"""
        start_time = time.time()
        
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on CPU usage
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            details = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": load_avg
            }
            
            return HealthResult(
                name="cpu_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthResult(
                name="cpu_usage",
                status=HealthStatus.UNHEALTHY,
                message="CPU health check failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def check_memory_health(self) -> HealthResult:
        """Check memory utilization"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on memory usage
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            details = {
                "memory_percent": memory.percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "swap_percent": swap.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2)
            }
            
            return HealthResult(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message="Memory health check failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def check_disk_health(self) -> HealthResult:
        """Check disk space utilization"""
        start_time = time.time()
        
        try:
            # Check disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on disk usage
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            details = {
                "disk_percent": round(disk_percent, 2),
                "disk_total_gb": round(disk_usage.total / (1024**3), 2),
                "disk_used_gb": round(disk_usage.used / (1024**3), 2),
                "disk_free_gb": round(disk_usage.free / (1024**3), 2)
            }
            
            return HealthResult(
                name="disk_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthResult(
                name="disk_usage",
                status=HealthStatus.UNHEALTHY,
                message="Disk health check failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

class WebSearchHealthChecker:
    """Web search service health checks"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
    
    async def check_web_search_health(self) -> HealthResult:
        """Check web search capabilities"""
        start_time = time.time()
        
        try:
            # Test basic internet connectivity
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Test DuckDuckGo (primary search provider)
                async with session.get("https://duckduckgo.com", 
                                     headers={'User-Agent': 'StudyForge-AI/1.0 Health Check'}) as response:
                    if response.status == 200:
                        response_time = (time.time() - start_time) * 1000
                        
                        return HealthResult(
                            name="web_search_connectivity",
                            status=HealthStatus.HEALTHY,
                            message="Web search connectivity operational",
                            timestamp=datetime.now(),
                            response_time_ms=response_time,
                            details={"provider": "duckduckgo", "status": "accessible"}
                        )
                    else:
                        return HealthResult(
                            name="web_search_connectivity",
                            status=HealthStatus.DEGRADED,
                            message=f"Search provider returned status {response.status}",
                            timestamp=datetime.now(),
                            response_time_ms=(time.time() - start_time) * 1000,
                            details={"provider": "duckduckgo", "status_code": response.status}
                        )
                        
        except Exception as e:
            return HealthResult(
                name="web_search_connectivity",
                status=HealthStatus.UNHEALTHY,
                message="Web search connectivity failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
        self.config = get_config()
        
        # Initialize health checkers
        self.db_checker = DatabaseHealthChecker()
        self.ai_checker = AIModelHealthChecker()
        self.system_checker = SystemResourceChecker()
        self.web_checker = WebSearchHealthChecker()
        
        # Health check registry
        self.health_checks: List[HealthCheck] = [
            HealthCheck("database", ComponentType.DATABASE, self.db_checker.check_sqlite_health, critical=True),
            HealthCheck("ai_model", ComponentType.AI_MODEL, self.ai_checker.check_ollama_health, critical=True),
            HealthCheck("cpu", ComponentType.CPU, self.system_checker.check_cpu_health),
            HealthCheck("memory", ComponentType.MEMORY, self.system_checker.check_memory_health),
            HealthCheck("disk", ComponentType.FILE_SYSTEM, self.system_checker.check_disk_health),
            HealthCheck("web_search", ComponentType.WEB_SEARCH, self.web_checker.check_web_search_health)
        ]
        
        # Results cache
        self.latest_results: Dict[str, HealthResult] = {}
        self.metrics_history: List[SystemMetrics] = []
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def run_single_health_check(self, check_name: str) -> Optional[HealthResult]:
        """Run a single health check by name"""
        for check in self.health_checks:
            if check.name == check_name and check.enabled:
                try:
                    result = await asyncio.wait_for(
                        check.check_function(),
                        timeout=check.timeout_seconds
                    )
                    self.latest_results[check_name] = result
                    return result
                except asyncio.TimeoutError:
                    result = HealthResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check timed out after {check.timeout_seconds}s",
                        timestamp=datetime.now(),
                        response_time_ms=check.timeout_seconds * 1000,
                        error="Timeout"
                    )
                    self.latest_results[check_name] = result
                    return result
                except Exception as e:
                    result = HealthResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message="Health check failed with exception",
                        timestamp=datetime.now(),
                        response_time_ms=0,
                        error=str(e)
                    )
                    self.latest_results[check_name] = result
                    return result
        return None
    
    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """Run all enabled health checks"""
        tasks = []
        check_names = []
        
        for check in self.health_checks:
            if check.enabled:
                tasks.append(self.run_single_health_check(check.name))
                check_names.append(check.name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = {}
        for name, result in zip(check_names, results):
            if isinstance(result, Exception):
                health_results[name] = HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Health check raised exception",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    error=str(result)
                )
            elif result:
                health_results[name] = result
        
        return health_results
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.latest_results:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": "No health checks available",
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
        
        # Determine overall status
        critical_failed = any(
            result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]
            for check in self.health_checks
            for name, result in self.latest_results.items()
            if check.name == name and check.critical
        )
        
        any_degraded = any(
            result.status == HealthStatus.DEGRADED
            for result in self.latest_results.values()
        )
        
        any_unhealthy = any(
            result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            for result in self.latest_results.values()
        )
        
        if critical_failed:
            overall_status = HealthStatus.CRITICAL
            message = "Critical components failing"
        elif any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
            message = "Some components unhealthy"
        elif any_degraded:
            overall_status = HealthStatus.DEGRADED
            message = "Some components degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.details,
                    "error": result.error
                }
                for name, result in self.latest_results.items()
            }
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Network
            network_io = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            
            # Connections (approximate)
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                connections = 0
            
            # Uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_connections=connections,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                disk_percent=0,
                network_io={},
                active_connections=0,
                uptime_seconds=0
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring dashboard"""
        current_metrics = self.get_system_metrics()
        
        # Calculate averages from history
        if self.metrics_history:
            recent_metrics = self.metrics_history[-60:]  # Last 60 data points
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = current_metrics.cpu_percent
            avg_memory = current_metrics.memory_percent
            avg_disk = current_metrics.disk_percent
        
        return {
            "current": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_percent": current_metrics.disk_percent,
                "active_connections": current_metrics.active_connections,
                "uptime_hours": round(current_metrics.uptime_seconds / 3600, 2)
            },
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_percent": round(avg_disk, 2)
            },
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count()
            },
            "timestamp": current_metrics.timestamp.isoformat()
        }
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring"""
        if self._running:
            return
        
        self._running = True
        self.logger.info(f"Starting health monitoring with {interval_seconds}s interval")
        
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self._running:
            try:
                # Run health checks
                await self.run_all_health_checks()
                
                # Collect system metrics
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours worth)
                max_metrics = int(86400 / interval_seconds)  # 24 hours
                if len(self.metrics_history) > max_metrics:
                    self.metrics_history = self.metrics_history[-max_metrics:]
                
                # Log health status
                overall_health = self.get_overall_health_status()
                self.logger.audit(
                    'health_check_completed',
                    'health_monitor',
                    overall_status=overall_health['status'],
                    checks_count=len(overall_health['checks']),
                    cpu_percent=metrics.cpu_percent,
                    memory_percent=metrics.memory_percent,
                    disk_percent=metrics.disk_percent
                )
                
                # Alert on critical issues
                if overall_health['status'] in ['critical', 'unhealthy']:
                    self.logger.critical(
                        f"System health {overall_health['status']}: {overall_health['message']}",
                        health_status=overall_health
                    )
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)

# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor