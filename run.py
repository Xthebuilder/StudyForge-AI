#!/usr/bin/env python3
"""
StudyForge AI - Web Application Launcher
Single command to start the web interface with enhanced features
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
from contextlib import contextmanager

# Third-party imports (will be checked and installed if needed)
try:
    import qrcode
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    qrcode = None
    colorama = None
    Fore = Style = type('MockColor', (), {'__getattr__': lambda self, name: ''})()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT_RANGE = (8000, 8999)
DEFAULT_HOST = '0.0.0.0'
REQUIRED_PACKAGES = {
    'fastapi': 'fastapi>=0.104.1',
    'uvicorn': 'uvicorn[standard]>=0.24.0',
    'jinja2': 'jinja2>=3.1.2',
    'aiohttp': 'aiohttp>=3.9.0',
    'qrcode': 'qrcode[pil]>=7.4.2',
    'colorama': 'colorama>=0.4.6'
}
REQUIRED_FILES = ['src', 'agent.py', 'web_server.py']
REQUIRED_DIRECTORIES = ['templates', 'static/css', 'static/js']

class DependencyManager:
    """Manages package dependencies with better error handling"""
    
    def __init__(self, packages: Dict[str, str]):
        self.packages = packages
        self.missing_packages = []
        self.failed_installs = []
    
    def check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        logger.info("Checking dependencies...")
        self.missing_packages.clear()
        
        for package_name in self.packages:
            try:
                __import__(package_name)
                logger.debug(f"✓ {package_name} is available")
            except ImportError:
                self.missing_packages.append(package_name)
                logger.debug(f"✗ {package_name} is missing")
        
        if self.missing_packages:
            print(f"{Fore.RED}❌ Missing required packages:{Style.RESET_ALL}")
            for package in self.missing_packages:
                print(f"   - {package}")
            return False
        
        print(f"{Fore.GREEN}✅ All dependencies are satisfied{Style.RESET_ALL}")
        return True
    
    def install_missing_packages(self, user_install: bool = True) -> bool:
        """Install missing packages with proper error handling"""
        if not self.missing_packages:
            return True
        
        print(f"\n{Fore.YELLOW}📦 Installing missing packages...{Style.RESET_ALL}")
        
        install_cmd = [sys.executable, "-m", "pip", "install"]
        if user_install:
            install_cmd.append("--user")
        
        # Install packages one by one for better error handling
        for package_name in self.missing_packages:
            package_spec = self.packages[package_name]
            print(f"Installing {package_spec}...")
            
            try:
                result = subprocess.run(
                    install_cmd + [package_spec],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if result.returncode == 0:
                    print(f"{Fore.GREEN}✅ {package_name} installed successfully{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Failed to install {package_name}: {result.stderr}{Style.RESET_ALL}")
                    self.failed_installs.append(package_name)
                    
            except subprocess.TimeoutExpired:
                print(f"{Fore.RED}❌ Timeout installing {package_name}{Style.RESET_ALL}")
                self.failed_installs.append(package_name)
            except Exception as e:
                print(f"{Fore.RED}❌ Error installing {package_name}: {e}{Style.RESET_ALL}")
                self.failed_installs.append(package_name)
        
        if self.failed_installs:
            print(f"\n{Fore.RED}❌ Failed to install some packages:{Style.RESET_ALL}")
            print("Please install manually with:")
            for package_name in self.failed_installs:
                package_spec = self.packages[package_name]
                print(f"  pip install {package_spec}")
            return False
        
        print(f"{Fore.GREEN}✅ All dependencies installed successfully!{Style.RESET_ALL}")
        return True

class NetworkManager:
    """Manages network-related functionality"""
    
    @staticmethod
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
        
        try:
            # Method 3: Network interfaces (Unix/Linux)
            if platform.system() != "Windows":
                result = subprocess.run(
                    ['ip', 'route', 'get', '8.8.8.8'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'src' in line:
                            local_ip = line.split('src')[1].strip().split()[0]
                            logger.debug(f"Local IP detected via ip route: {local_ip}")
                            return local_ip
        except Exception as e:
            logger.debug(f"IP route method failed: {e}")
        
        logger.warning("Could not detect local IP, using localhost")
        return "127.0.0.1"

    @staticmethod
    def is_port_available(port: int, host: str = '0.0.0.0') -> bool:
        """Check if a port is available on the given host"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except (OSError, PermissionError) as e:
            logger.debug(f"Port {port} not available: {e}")
            return False
    
    @staticmethod
    def find_available_port(start_port: int = 8000, max_port: int = 8999, host: str = '0.0.0.0') -> Optional[int]:
        """Find an available port starting from start_port"""
        logger.info(f"Searching for available port in range {start_port}-{max_port}")
        
        for port in range(start_port, max_port + 1):
            if NetworkManager.is_port_available(port, host):
                logger.info(f"Found available port: {port}")
                return port
        
        logger.error(f"No available ports found in range {start_port}-{max_port}")
        return None
    
    @staticmethod
    def get_qr_code_terminal(url: str) -> Optional[str]:
        """Generate QR code for terminal display"""
        if not qrcode:
            logger.warning("QR code library not available")
            return None
        
        try:
            qr = qrcode.QRCode(version=1, box_size=1, border=1)
            qr.add_data(url)
            qr.make(fit=True)
            
            # Capture QR code as string
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                qr.print_ascii(invert=True)
            
            return output.getvalue()
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            return None


class SystemChecker:
    """System and service checking utilities"""
    
    @staticmethod
    def check_python_version() -> bool:
        """Check if Python version is supported"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            print(f"{Fore.RED}❌ Python {'.'.join(map(str, min_version))}+ required, found {'.'.join(map(str, current_version))}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}✅ Python {'.'.join(map(str, current_version))} detected{Style.RESET_ALL}")
        return True
    
    @staticmethod
    def check_directory_structure() -> bool:
        """Check if we're in the correct directory with required files"""
        current_dir = Path.cwd()
        logger.info(f"Checking directory structure in: {current_dir}")
        
        missing_files = []
        for required_file in REQUIRED_FILES:
            if not Path(required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            print(f"{Fore.RED}❌ Missing required files/directories:{Style.RESET_ALL}")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPlease run this script from the StudyForge-AI directory")
            return False
        
        print(f"{Fore.GREEN}✅ Directory structure validated{Style.RESET_ALL}")
        return True
    
    @staticmethod
    def create_required_directories() -> None:
        """Create required directories if they don't exist"""
        logger.info("Creating required directories...")
        
        for directory in REQUIRED_DIRECTORIES:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    @staticmethod
    async def check_ollama_connection() -> bool:
        """Check if Ollama is running and accessible"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'http://localhost:11434/api/tags',
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.debug(f"Ollama connection check failed: {e}")
            return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='StudyForge AI Web Interface Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run.py                    # Start with default settings
  python run.py --port 8080        # Use specific port
  python run.py --no-browser       # Don't open browser
  python run.py --host 127.0.0.1   # Bind to localhost only
  python run.py --dev              # Development mode with reload
'''
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Port to run the server on (default: find available port 8000-8999)'
    )
    
    parser.add_argument(
        '--host',
        default=DEFAULT_HOST,
        help=f'Host to bind to (default: {DEFAULT_HOST})'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run in development mode with auto-reload'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency checking and installation'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Reduce noise from uvicorn in non-verbose mode
    if not verbose:
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file"""
    default_config = {
        'host': DEFAULT_HOST,
        'port_range': DEFAULT_PORT_RANGE,
        'auto_open_browser': True,
        'development_mode': False
    }
    
    if not config_path:
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = {**default_config, **user_config}
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return default_config


@contextmanager
def browser_opener(url: str, delay: float = 2.0):
    """Context manager to open browser in background thread"""
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            logger.info(f"Opened browser to: {url}")
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
    
    thread = threading.Thread(target=open_browser, daemon=True)
    thread.start()
    try:
        yield
    finally:
        pass  # Thread is daemon, will die with main process


def check_ollama_sync() -> bool:
    """Synchronous wrapper for Ollama connection check"""
    try:
        return asyncio.run(SystemChecker.check_ollama_connection())
    except Exception:
        return False

def main():
    """Enhanced main launcher function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"{Fore.CYAN}🚀 StudyForge AI - Web Interface Launcher{Style.RESET_ALL}")
    print(f"{Fore.CYAN}" + "=" * 50 + f"{Style.RESET_ALL}")
    print(f"Python: {platform.python_version()} on {platform.system()}")
    print(f"Working directory: {Path.cwd()}")
    print()
    
    try:
        # System checks
        print("🔍 Running system checks...")
        if not SystemChecker.check_python_version():
            sys.exit(1)
        
        if not SystemChecker.check_directory_structure():
            sys.exit(1)
        
        SystemChecker.create_required_directories()
        
        # Check dependencies unless skipped
        if not args.skip_deps:
            print("\n📦 Checking dependencies...")
            dep_manager = DependencyManager(REQUIRED_PACKAGES)
            
            if not dep_manager.check_dependencies():
                print("\n❓ Would you like to install missing packages? (y/n): ", end="")
                try:
                    response = input().strip().lower()
                    
                    if response in ('y', 'yes'):
                        if not dep_manager.install_missing_packages():
                            print(f"\n{Fore.RED}❌ Failed to install some dependencies{Style.RESET_ALL}")
                            sys.exit(1)
                    else:
                        print(f"\n{Fore.YELLOW}⚠️  Continuing without installing missing packages...{Style.RESET_ALL}")
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{Fore.YELLOW}⚠️  Continuing without installing missing packages...{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⏭️  Skipping dependency check...{Style.RESET_ALL}")
        
        # Check Ollama connection
        print("\n🔗 Checking Ollama connection...")
        if check_ollama_sync():
            print(f"{Fore.GREEN}✅ Ollama is running and accessible{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️  Ollama not detected. Make sure Ollama is running:{Style.RESET_ALL}")
            print("   - Install Ollama: https://ollama.ai/")
            print("   - Start Ollama: ollama serve")
            print("   - Pull a model: ollama pull llama3:8b")
            print("\nContinuing anyway... (AI features may not work)")
        
        # Determine port
        port = args.port
        if not port:
            port_range = config.get('port_range', DEFAULT_PORT_RANGE)
            port = NetworkManager.find_available_port(port_range[0], port_range[1], args.host)
            if not port:
                print(f"{Fore.RED}❌ No available ports found in range {port_range[0]}-{port_range[1]}{Style.RESET_ALL}")
                sys.exit(1)
        elif not NetworkManager.is_port_available(port, args.host):
            print(f"{Fore.RED}❌ Port {port} is not available{Style.RESET_ALL}")
            sys.exit(1)
        
        # Get network info
        local_ip = NetworkManager.get_local_ip()
        local_url = f"http://127.0.0.1:{port}"
        lan_url = f"http://{local_ip}:{port}"
        
        # Display startup information
        print(f"\n🌐 Starting StudyForge AI Web Server...")
        print(f"📍 Local Access:  {local_url}")
        print(f"🌍 LAN Access:    {lan_url}")
        print(f"📱 Mobile Access: Scan QR code below")
        
        # Display QR code
        qr_code = NetworkManager.get_qr_code_terminal(lan_url)
        if qr_code:
            print("\n📱 QR Code for Mobile Access:")
            print(qr_code)
        else:
            print(f"\n📱 Mobile URL: {lan_url}")
            print("(Install 'qrcode' package for QR code display)")
        
        print(f"\n🎯 Features Available:")
        print("  ✅ ChatGPT-style interface")
        print("  ✅ Real-time web search integration")
        print("  ✅ Session management & history")
        print("  ✅ Mobile-responsive design")
        print("  ✅ Dark/Light theme support")
        print("  ✅ LAN accessibility")
        print("  ✅ WebSocket real-time communication")
        
        print(f"\n🛠️  Configuration:")
        print(f"  - Port: {port}")
        print(f"  - Host: {args.host}")
        print(f"  - Database: SQLite (sessions.db)")
        print(f"  - Development mode: {args.dev}")
        
        print("\n⌨️  Controls:")
        print("  - Press Ctrl+C to stop the server")
        if not args.no_browser and not args.dev:
            print("  - Server will auto-open in your browser")
        print("=" * 50)
        
        # Open browser if requested
        if not args.no_browser and not args.dev:
            with browser_opener(local_url):
                pass  # Context manager handles browser opening
        
        # Import and run the web server
        try:
            import uvicorn
        except ImportError:
            print(f"{Fore.RED}❌ uvicorn not found. Please install with: pip install uvicorn[standard]{Style.RESET_ALL}")
            sys.exit(1)
        
        # Configure uvicorn settings
        uvicorn_config = {
            "app": "web_server:app",
            "host": args.host,
            "port": port,
            "reload": args.dev,
            "access_log": args.verbose,
            "log_level": "debug" if args.verbose else "info"
        }
        
        # Run the server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.CYAN}👋 StudyForge AI Web Server stopped.{Style.RESET_ALL}")
        print("Thanks for using StudyForge AI!")
        
    except Exception as e:
        logger.exception("Unexpected error in main()")
        print(f"\n{Fore.RED}❌ Error starting server: {e}{Style.RESET_ALL}")
        print("Please check the error message and try again.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.exception("Failed to start StudyForge AI")
        sys.exit(1)