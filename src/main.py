#!/usr/bin/env python3
"""
Timeout-Enhanced AI Agent
Demonstrates advanced timeout handling for long-running AI tasks
"""

import json
import sqlite3
import requests
import asyncio
import aiohttp
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import threading
import time

# Color support
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    class MockColor:
        def __getattr__(self, name): return ""
    Fore = Style = MockColor()


@dataclass
class TimeoutConfig:
    """Configuration for various timeout scenarios"""
    # Request timeouts (in seconds)
    connection_timeout: int = 10      # Time to establish connection
    read_timeout: int = 300           # Time to wait for response (5 minutes)
    total_timeout: int = 600          # Total request timeout (10 minutes)
    
    # Session timeouts
    idle_timeout: int = 1800          # 30 minutes of inactivity
    max_session_duration: int = 7200  # 2 hours max session
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 5
    
    # Long-running task handling
    progress_update_interval: int = 30  # Update every 30 seconds
    heartbeat_interval: int = 60        # Heartbeat every minute


class TimeoutManager:
    """Manages timeouts and keeps sessions alive"""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.last_activity = datetime.now()
        self.session_start = datetime.now()
        self.is_active = True
        self._heartbeat_thread = None
        self._setup_heartbeat()
    
    def _setup_heartbeat(self):
        """Setup heartbeat to prevent timeouts"""
        def heartbeat():
            while self.is_active:
                time.sleep(self.config.heartbeat_interval)
                if self.is_active:
                    print(f"{Fore.CYAN}💓 Session heartbeat - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
                    self.update_activity()
        
        self._heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self._heartbeat_thread.start()
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def check_session_limits(self) -> bool:
        """Check if session limits are exceeded"""
        now = datetime.now()
        
        # Check idle timeout
        idle_duration = (now - self.last_activity).total_seconds()
        if idle_duration > self.config.idle_timeout:
            print(f"{Fore.YELLOW}⚠️  Session idle for {idle_duration/60:.1f} minutes{Style.RESET_ALL}")
            return False
        
        # Check max session duration
        session_duration = (now - self.session_start).total_seconds()
        if session_duration > self.config.max_session_duration:
            print(f"{Fore.YELLOW}⚠️  Session running for {session_duration/3600:.1f} hours{Style.RESET_ALL}")
            return False
        
        return True
    
    def get_remaining_time(self) -> Dict[str, float]:
        """Get remaining time for various limits"""
        now = datetime.now()
        idle_remaining = self.config.idle_timeout - (now - self.last_activity).total_seconds()
        session_remaining = self.config.max_session_duration - (now - self.session_start).total_seconds()
        
        return {
            "idle_remaining_minutes": max(0, idle_remaining / 60),
            "session_remaining_minutes": max(0, session_remaining / 60)
        }
    
    def shutdown(self):
        """Gracefully shutdown timeout manager"""
        self.is_active = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1)


class TimeoutAwareHTTPClient:
    """HTTP client with comprehensive timeout handling"""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.session = None
    
    async def create_session(self):
        """Create aiohttp session with timeout configuration"""
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.read_timeout
        )
        
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def post_with_retry(self, url: str, data: Dict[str, Any], 
                             progress_callback=None) -> Dict[str, Any]:
        """Make POST request with retry logic and progress updates"""
        if not self.session:
            await self.create_session()
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                print(f"{Fore.CYAN}🔄 Attempt {attempt + 1}/{self.config.max_retries} - Connecting to {url}{Style.RESET_ALL}")
                
                # Start progress monitoring for long requests
                progress_task = None
                if progress_callback:
                    progress_task = asyncio.create_task(self._monitor_progress(progress_callback))
                
                async with self.session.post(url, json=data) as response:
                    if progress_task:
                        progress_task.cancel()
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"{Fore.GREEN}✅ Request completed successfully{Style.RESET_ALL}")
                        return result
                    else:
                        print(f"{Fore.RED}❌ HTTP {response.status}: {response.reason}{Style.RESET_ALL}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
            except asyncio.TimeoutError as e:
                last_exception = e
                print(f"{Fore.YELLOW}⏱️  Request timed out on attempt {attempt + 1}{Style.RESET_ALL}")
                
            except aiohttp.ClientError as e:
                last_exception = e
                print(f"{Fore.RED}🔌 Connection error on attempt {attempt + 1}: {str(e)}{Style.RESET_ALL}")
            
            except Exception as e:
                last_exception = e
                print(f"{Fore.RED}❌ Unexpected error on attempt {attempt + 1}: {str(e)}{Style.RESET_ALL}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_delay * (attempt + 1)  # Exponential backoff
                print(f"{Fore.CYAN}⏳ Waiting {wait_time} seconds before retry...{Style.RESET_ALL}")
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        print(f"{Fore.RED}💀 All {self.config.max_retries} attempts failed{Style.RESET_ALL}")
        raise last_exception if last_exception else Exception("All retry attempts failed")
    
    async def _monitor_progress(self, callback):
        """Monitor long-running request progress"""
        elapsed = 0
        while True:
            await asyncio.sleep(self.config.progress_update_interval)
            elapsed += self.config.progress_update_interval
            callback(elapsed)
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()


class TimeoutEnhancedAI:
    """Main AI agent with comprehensive timeout handling"""
    
    def __init__(self):
        self.config = TimeoutConfig()
        self.timeout_manager = TimeoutManager(self.config)
        self.http_client = TimeoutAwareHTTPClient(self.config)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "gpt-oss:20b"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"{Fore.GREEN}🤖 Timeout-Enhanced AI Agent initialized{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⏱️  Timeout Configuration:{Style.RESET_ALL}")
        print(f"   • Connection timeout: {self.config.connection_timeout}s")
        print(f"   • Read timeout: {self.config.read_timeout}s ({self.config.read_timeout/60:.1f} minutes)")
        print(f"   • Total timeout: {self.config.total_timeout}s ({self.config.total_timeout/60:.1f} minutes)")
        print(f"   • Session limit: {self.config.max_session_duration/3600:.1f} hours")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n{Fore.YELLOW}🛑 Received shutdown signal, cleaning up...{Style.RESET_ALL}")
        asyncio.create_task(self.shutdown())
    
    def progress_callback(self, elapsed_seconds: int):
        """Callback for long-running request progress"""
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        print(f"{Fore.MAGENTA}⏳ Request in progress: {minutes:02d}:{seconds:02d} elapsed{Style.RESET_ALL}")
    
    async def query_ollama(self, prompt: str, context: str = "") -> str:
        """Query Ollama with timeout handling"""
        self.timeout_manager.update_activity()
        
        # Check session limits
        if not self.timeout_manager.check_session_limits():
            remaining = self.timeout_manager.get_remaining_time()
            print(f"{Fore.RED}⚠️  Session limits exceeded{Style.RESET_ALL}")
            print(f"Idle time remaining: {remaining['idle_remaining_minutes']:.1f} minutes")
            print(f"Session time remaining: {remaining['session_remaining_minutes']:.1f} minutes")
            return "Session timeout reached. Please restart the application."
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        try:
            print(f"{Fore.BLUE}🧠 Processing your request with {self.model_name}...{Style.RESET_ALL}")
            
            response = await self.http_client.post_with_retry(
                self.ollama_url, 
                data, 
                progress_callback=self.progress_callback
            )
            
            if 'response' in response:
                return response['response']
            else:
                return "I received an unexpected response format from the AI model."
                
        except asyncio.TimeoutError:
            return f"⏱️ Request timed out after {self.config.total_timeout/60:.1f} minutes. The AI model might be processing a complex request. Please try a simpler question or restart the service."
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error querying AI: {str(e)}{Style.RESET_ALL}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def print_session_status(self):
        """Print current session status"""
        remaining = self.timeout_manager.get_remaining_time()
        
        print(f"\n{Fore.CYAN}📊 Session Status:{Style.RESET_ALL}")
        print(f"   • Idle timeout: {remaining['idle_remaining_minutes']:.1f} minutes remaining")
        print(f"   • Session timeout: {remaining['session_remaining_minutes']:.1f} minutes remaining")
        print(f"   • Last activity: {self.timeout_manager.last_activity.strftime('%H:%M:%S')}")
        print(f"   • Session started: {self.timeout_manager.session_start.strftime('%H:%M:%S')}")
    
    async def run_chat_loop(self):
        """Main chat loop with timeout awareness"""
        print(f"\n{Fore.GREEN}🎓 Welcome to Timeout-Enhanced AI Agent!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This agent demonstrates advanced timeout handling:{Style.RESET_ALL}")
        print("• Long requests won't timeout unexpectedly")
        print("• Session keeps alive with heartbeats")
        print("• Automatic retry on connection issues")
        print("• Progress updates during long operations")
        print(f"{Fore.CYAN}Type '/status' to see session info, '/quit' to exit{Style.RESET_ALL}")
        
        try:
            while True:
                # Check session limits before each interaction
                if not self.timeout_manager.check_session_limits():
                    print(f"{Fore.RED}Session limits exceeded. Exiting...{Style.RESET_ALL}")
                    break
                
                try:
                    user_input = input(f"\n{Fore.LIGHTCYAN_EX}You: {Style.RESET_ALL}").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                        break
                    
                    if user_input.lower() == '/status':
                        self.print_session_status()
                        continue
                    
                    if user_input.lower() == '/timeout-test':
                        print(f"{Fore.YELLOW}🧪 Running timeout test with long prompt...{Style.RESET_ALL}")
                        test_prompt = "Write a detailed explanation of quantum computing, including mathematical foundations, practical applications, and future prospects. Include code examples and diagrams where appropriate." * 5
                        response = await self.query_ollama(test_prompt)
                    else:
                        response = await self.query_ollama(user_input)
                    
                    print(f"\n{Fore.LIGHTGREEN_EX}AI: {Style.RESET_ALL}{response}")
                    
                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}Received interrupt, shutting down gracefully...{Style.RESET_ALL}")
                    break
                except EOFError:
                    print(f"\n{Fore.YELLOW}Input stream closed, shutting down...{Style.RESET_ALL}")
                    break
                except Exception as e:
                    print(f"{Fore.RED}❌ Error in chat loop: {str(e)}{Style.RESET_ALL}")
                    
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        print(f"{Fore.CYAN}🛑 Shutting down Timeout-Enhanced AI Agent...{Style.RESET_ALL}")
        
        # Shutdown components
        self.timeout_manager.shutdown()
        await self.http_client.close()
        
        print(f"{Fore.GREEN}✅ Shutdown complete. Goodbye!{Style.RESET_ALL}")


async def main():
    """Main entry point"""
    ai = TimeoutEnhancedAI()
    await ai.run_chat_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")