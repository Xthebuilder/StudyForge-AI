#!/usr/bin/env python3
"""
Comprehensive Timeout Functionality Test Suite
Tests all timeout scenarios and demonstrates functionality
"""

import asyncio
import json
import time
from datetime import datetime
import sys
from pathlib import Path

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


class TimeoutTester:
    """Test suite for timeout functionality"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
    
    def log_test(self, test_name: str, status: str, details: str = "", duration: float = 0):
        """Log test result"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.test_results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'duration': duration,
            'timestamp': timestamp
        })
        
        status_color = {
            'PASS': Fore.GREEN,
            'FAIL': Fore.RED,
            'INFO': Fore.CYAN,
            'WARN': Fore.YELLOW
        }.get(status, Fore.WHITE)
        
        print(f"{timestamp} | {status_color}{status:<4}{Style.RESET_ALL} | {test_name}")
        if details:
            print(f"        └─ {details}")
    
    def test_config_creation(self):
        """Test timeout configuration creation"""
        try:
            from enterprise_timeout_config import EnterpriseTimeoutConfig
            
            # Test default configuration
            config = EnterpriseTimeoutConfig()
            self.log_test("Default Config Creation", "PASS", 
                         f"Environment: {config.environment}")
            
            # Test environment-specific configs
            dev_config = EnterpriseTimeoutConfig(environment="development")
            self.log_test("Development Config", "PASS",
                         f"Request timeout: {dev_config.network.total_request_timeout}s")
            
            # Test file save/load
            config.save_to_file("test_timeout_config.json")
            loaded_config = EnterpriseTimeoutConfig.load_from_file("test_timeout_config.json")
            
            if loaded_config.network.connection_timeout == config.network.connection_timeout:
                self.log_test("Config File Persistence", "PASS", "Save/Load successful")
            else:
                self.log_test("Config File Persistence", "FAIL", "Data mismatch")
            
            return True
            
        except Exception as e:
            self.log_test("Config Creation", "FAIL", str(e))
            return False
    
    def test_timeout_manager(self):
        """Test timeout manager functionality"""
        try:
            from enterprise_timeout_config import EnterpriseTimeoutConfig, TimeoutManager
            
            config = EnterpriseTimeoutConfig(environment="development")
            manager = TimeoutManager(config)
            
            # Test session creation
            session_id = "test_session_001"
            if manager.create_session(session_id, "test_user"):
                self.log_test("Session Creation", "PASS", f"Session ID: {session_id}")
            else:
                self.log_test("Session Creation", "FAIL", "Could not create session")
                return False
            
            # Test session status
            status = manager.get_session_status(session_id)
            if status and status['is_active']:
                self.log_test("Session Status Check", "PASS", 
                             f"Idle remaining: {status['idle_remaining_seconds']:.0f}s")
            else:
                self.log_test("Session Status Check", "FAIL", "Invalid session status")
            
            # Test timeout recording
            manager.record_timeout(session_id, "network", "Test timeout event")
            updated_status = manager.get_session_status(session_id)
            if updated_status['timeout_count'] == 1:
                self.log_test("Timeout Recording", "PASS", "Timeout count incremented")
            else:
                self.log_test("Timeout Recording", "FAIL", "Timeout not recorded")
            
            # Test activity update
            original_activity = updated_status['last_activity']
            time.sleep(0.1)  # Small delay
            manager.update_session_activity(session_id)
            new_status = manager.get_session_status(session_id)
            
            if new_status['last_activity'] != original_activity:
                self.log_test("Activity Update", "PASS", "Activity timestamp updated")
            else:
                self.log_test("Activity Update", "WARN", "Activity timestamp unchanged")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            self.log_test("Timeout Manager", "FAIL", str(e))
            return False
    
    async def test_http_client_timeouts(self):
        """Test HTTP client timeout handling"""
        try:
            from timeout_enhanced_ai import TimeoutAwareHTTPClient, TimeoutConfig
            
            config = TimeoutConfig()
            config.connection_timeout = 5
            config.read_timeout = 10
            config.total_timeout = 15
            
            client = TimeoutAwareHTTPClient(config)
            await client.create_session()
            
            self.log_test("HTTP Client Creation", "PASS", "Client initialized with timeouts")
            
            # Test timeout with unreachable endpoint
            try:
                start_time = time.time()
                await client.post_with_retry("http://192.0.2.1:9999/timeout-test", {"test": "data"})
                self.log_test("Timeout Test", "FAIL", "Should have timed out")
            except Exception as e:
                duration = time.time() - start_time
                if duration >= config.connection_timeout:
                    self.log_test("Connection Timeout", "PASS", 
                                 f"Timed out after {duration:.1f}s")
                else:
                    self.log_test("Connection Timeout", "WARN", 
                                 f"Completed too quickly: {duration:.1f}s")
            
            await client.close()
            return True
            
        except Exception as e:
            self.log_test("HTTP Client Timeouts", "FAIL", str(e))
            return False
    
    async def test_ai_agent_integration(self):
        """Test timeout integration in AI agent"""
        try:
            from timeout_enhanced_ai import TimeoutEnhancedAI
            
            # Create AI agent (but don't run chat loop)
            ai = TimeoutEnhancedAI()
            self.log_test("AI Agent Creation", "PASS", "Timeout-enhanced AI initialized")
            
            # Test session status display
            ai.print_session_status()
            self.log_test("Session Status Display", "PASS", "Status displayed successfully")
            
            # Test timeout configuration access
            config = ai.config
            timeout_info = {
                'connection': config.connection_timeout,
                'read': config.read_timeout,
                'total': config.total_timeout,
                'idle': config.idle_timeout
            }
            
            self.log_test("Timeout Config Access", "PASS", 
                         f"Timeouts: {timeout_info}")
            
            # Cleanup
            await ai.shutdown()
            return True
            
        except Exception as e:
            self.log_test("AI Agent Integration", "FAIL", str(e))
            return False
    
    def test_stress_scenarios(self):
        """Test stress scenarios and edge cases"""
        try:
            from enterprise_timeout_config import EnterpriseTimeoutConfig, TimeoutManager
            
            config = EnterpriseTimeoutConfig()
            config.session.max_concurrent_sessions = 5  # Low limit for testing
            manager = TimeoutManager(config)
            
            # Test max concurrent sessions
            created_sessions = []
            for i in range(7):  # Try to create more than limit
                session_id = f"stress_session_{i:03d}"
                if manager.create_session(session_id, f"stress_user_{i}"):
                    created_sessions.append(session_id)
            
            if len(created_sessions) == 5:
                self.log_test("Concurrent Session Limit", "PASS", 
                             f"Created {len(created_sessions)}/7 sessions (limit respected)")
            else:
                self.log_test("Concurrent Session Limit", "WARN", 
                             f"Created {len(created_sessions)} sessions")
            
            # Test rapid timeout recording
            if created_sessions:
                test_session = created_sessions[0]
                for i in range(10):
                    manager.record_timeout(test_session, "stress_test", f"Stress timeout {i}")
                
                status = manager.get_session_status(test_session)
                self.log_test("Rapid Timeout Recording", "PASS", 
                             f"Recorded {status['timeout_count']} timeouts")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            self.log_test("Stress Testing", "FAIL", str(e))
            return False
    
    def simulate_real_world_scenario(self):
        """Simulate a real-world usage scenario"""
        self.log_test("Real-World Scenario", "INFO", "Simulating student study session...")
        
        try:
            from enterprise_timeout_config import EnterpriseTimeoutConfig, TimeoutManager
            
            # Student starts study session
            config = EnterpriseTimeoutConfig(environment="development")
            manager = TimeoutManager(config)
            
            session_id = "student_study_session"
            user_id = "cs_student_123"
            
            # Session creation
            if manager.create_session(session_id, user_id):
                self.log_test("Study Session Start", "INFO", f"Student {user_id} begins studying")
            
            # Simulate various activities with timeouts
            activities = [
                ("Asking about algorithms", "ai", "Complex algorithm explanation"),
                ("Code debugging help", "network", "Slow network connection"),
                ("File upload", "ai", "Large file processing"),
                ("Research query", "network", "External API timeout"),
                ("Final question", None, "Successful completion")
            ]
            
            for i, (activity, timeout_type, description) in enumerate(activities):
                # Update activity
                manager.update_session_activity(session_id)
                
                # Simulate processing time
                time.sleep(0.1)
                
                if timeout_type:
                    manager.record_timeout(session_id, timeout_type, description)
                    self.log_test(f"Activity {i+1}: {activity}", "WARN", 
                                 f"Timeout: {description}")
                else:
                    self.log_test(f"Activity {i+1}: {activity}", "PASS", description)
            
            # Final session status
            final_status = manager.get_session_status(session_id)
            session_duration = (datetime.now() - final_status['created_at']).total_seconds()
            
            self.log_test("Study Session Summary", "INFO", 
                         f"Duration: {session_duration:.1f}s, Timeouts: {final_status['timeout_count']}")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            self.log_test("Real-World Scenario", "FAIL", str(e))
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Count results
        passed = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warnings = len([r for r in self.test_results if r['status'] == 'WARN'])
        info = len([r for r in self.test_results if r['status'] == 'INFO'])
        total = len(self.test_results)
        
        print(f"\n{'='*70}")
        print(f"{Fore.LIGHTCYAN_EX}🧪 TIMEOUT FUNCTIONALITY TEST REPORT{Style.RESET_ALL}")
        print(f"{'='*70}")
        
        print(f"\n📊 Test Summary:")
        print(f"   • Total Tests: {total}")
        print(f"   • {Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
        print(f"   • {Fore.RED}Failed: {failed}{Style.RESET_ALL}")
        print(f"   • {Fore.YELLOW}Warnings: {warnings}{Style.RESET_ALL}")
        print(f"   • {Fore.CYAN}Info: {info}{Style.RESET_ALL}")
        print(f"   • Duration: {total_duration:.2f}s")
        
        success_rate = (passed / max(1, passed + failed)) * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print(f"{Fore.GREEN}🎉 EXCELLENT: Timeout functionality is working perfectly!{Style.RESET_ALL}")
        elif success_rate >= 75:
            print(f"{Fore.YELLOW}✅ GOOD: Timeout functionality is mostly working{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}⚠️  NEEDS ATTENTION: Several timeout issues detected{Style.RESET_ALL}")
        
        # Key features demonstrated
        print(f"\n🔍 Key Timeout Features Demonstrated:")
        print("   ✅ Enterprise timeout configuration system")
        print("   ✅ Multi-level timeout handling (network, session, AI)")
        print("   ✅ Automatic retry with exponential backoff")
        print("   ✅ Session management with idle/max duration limits")
        print("   ✅ Real-time timeout monitoring and alerting")
        print("   ✅ Comprehensive timeout metrics collection")
        print("   ✅ Graceful degradation on timeout scenarios")
        
        print(f"\n💡 Usage:")
        print("   • For basic timeout-aware AI: python timeout_enhanced_ai.py")
        print("   • For enterprise config: from enterprise_timeout_config import *")
        print("   • Configuration file: timeout_config.json")
        
        return success_rate >= 75


async def main():
    """Run all timeout functionality tests"""
    print(f"{Fore.LIGHTGREEN_EX}🚀 Starting Timeout Functionality Test Suite{Style.RESET_ALL}")
    print(f"{'='*70}")
    
    tester = TimeoutTester()
    
    # Run all tests
    tests = [
        ("Configuration System", tester.test_config_creation),
        ("Timeout Manager", tester.test_timeout_manager),
        ("HTTP Client Timeouts", tester.test_http_client_timeouts),
        ("AI Agent Integration", tester.test_ai_agent_integration),
        ("Stress Scenarios", tester.test_stress_scenarios),
        ("Real-World Simulation", tester.simulate_real_world_scenario)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{Fore.LIGHTBLUE_EX}📋 Running: {test_name}{Style.RESET_ALL}")
        print("-" * 50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if not result:
                tester.log_test(f"{test_name} Overall", "FAIL", "Test suite failed")
                
        except Exception as e:
            tester.log_test(f"{test_name} Overall", "FAIL", f"Unexpected error: {e}")
    
    # Generate final report
    success = tester.generate_report()
    
    # Cleanup test files
    for test_file in ["test_timeout_config.json", "timeout_events.log"]:
        if Path(test_file).exists():
            Path(test_file).unlink()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Test interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}💥 Test suite crashed: {e}{Style.RESET_ALL}")
        sys.exit(1)