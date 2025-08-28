#!/usr/bin/env python3
"""
StudyForge AI - Usage Examples
Demonstrates various ways to use the StudyForge AI system
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enterprise_timeout_config import EnterpriseTimeoutConfig, TimeoutManager


def example_basic_usage():
    """Example: Basic StudyForge AI usage"""
    print("🎓 Example 1: Basic StudyForge AI Usage")
    print("=" * 50)
    
    print("""
    # Start StudyForge AI
    python src/main.py
    
    # Example conversation:
    You: Help me understand binary search algorithms
    AI: Binary search is a divide-and-conquer algorithm...
    
    You: Can you show me a Python implementation?  
    AI: Here's a clean Python implementation of binary search...
    
    You: What's the time complexity?
    AI: Binary search has O(log n) time complexity because...
    """)


def example_timeout_configuration():
    """Example: Custom timeout configuration"""
    print("\n⏱️  Example 2: Custom Timeout Configuration")
    print("=" * 50)
    
    # Create development configuration with longer timeouts
    config = EnterpriseTimeoutConfig(environment="development")
    
    print(f"📊 Default Development Settings:")
    print(f"   • Connection timeout: {config.network.connection_timeout}s")
    print(f"   • Request timeout: {config.network.total_request_timeout}s ({config.network.total_request_timeout/60:.0f} min)")
    print(f"   • Session idle timeout: {config.session.idle_timeout}s ({config.session.idle_timeout/60:.0f} min)")
    print(f"   • Max session duration: {config.session.max_session_duration}s ({config.session.max_session_duration/3600:.1f} hours)")
    
    # Customize for specific needs
    config.network.total_request_timeout = 1800  # 30 minutes for complex queries
    config.ai_model.inference_timeout = 1200     # 20 minutes for AI processing
    config.session.idle_timeout = 7200           # 2 hours idle time
    
    print(f"\n🔧 Customized Settings:")
    print(f"   • Request timeout: {config.network.total_request_timeout}s ({config.network.total_request_timeout/60:.0f} min)")
    print(f"   • AI inference timeout: {config.ai_model.inference_timeout}s ({config.ai_model.inference_timeout/60:.0f} min)")
    print(f"   • Session idle timeout: {config.session.idle_timeout}s ({config.session.idle_timeout/3600:.1f} hours)")
    
    # Save configuration
    config.save_to_file("custom_timeout_config.json")
    print(f"✅ Configuration saved to custom_timeout_config.json")


def example_session_management():
    """Example: Session management and monitoring"""
    print("\n👤 Example 3: Session Management")
    print("=" * 50)
    
    # Create timeout manager
    config = EnterpriseTimeoutConfig(environment="development")
    manager = TimeoutManager(config)
    
    # Create a study session
    session_id = "study_session_cs101"
    user_id = "student_alice"
    
    if manager.create_session(session_id, user_id):
        print(f"✅ Created study session: {session_id}")
        
        # Get session status
        status = manager.get_session_status(session_id)
        print(f"📊 Session Status:")
        print(f"   • User: {status['user_id']}")
        print(f"   • Created: {status['created_at']}")
        print(f"   • Idle remaining: {status['idle_remaining_seconds']:.0f} seconds")
        print(f"   • Session remaining: {status['session_remaining_seconds']:.0f} seconds")
        print(f"   • Is active: {status['is_active']}")
        
        # Simulate some activity
        manager.update_session_activity(session_id)
        print(f"✅ Updated session activity")
        
        # Simulate a timeout event (for demonstration)
        manager.record_timeout(session_id, "network", "Simulated connection timeout")
        print(f"⚠️  Recorded timeout event (simulation)")
        
        # Check updated status
        updated_status = manager.get_session_status(session_id)
        print(f"📊 Updated Status:")
        print(f"   • Timeout count: {updated_status['timeout_count']}")
    
    manager.shutdown()
    print(f"🛑 Session manager shutdown")


def example_production_deployment():
    """Example: Production deployment configuration"""
    print("\n🚀 Example 4: Production Deployment")
    print("=" * 50)
    
    # Production configuration
    prod_config = EnterpriseTimeoutConfig(environment="production")
    
    print(f"🏭 Production Configuration:")
    print(f"   • Environment: {prod_config.environment}")
    print(f"   • Connection timeout: {prod_config.network.connection_timeout}s")
    print(f"   • Request timeout: {prod_config.network.total_request_timeout}s")
    print(f"   • Max concurrent sessions: {prod_config.session.max_concurrent_sessions}")
    print(f"   • Monitoring enabled: {prod_config.monitoring.enable_metrics}")
    print(f"   • Alert on repeated timeouts: {prod_config.monitoring.alert_on_repeated_timeouts}")
    
    print(f"\n🔧 Production Best Practices:")
    print(f"   • Use environment variables for sensitive config")
    print(f"   • Enable comprehensive logging")
    print(f"   • Set up monitoring and alerting")
    print(f"   • Configure load balancing for multiple instances")
    print(f"   • Implement health checks")
    
    print(f"\n📋 Deployment Checklist:")
    print(f"   ✅ Ollama service running and accessible")
    print(f"   ✅ Required AI models downloaded")
    print(f"   ✅ Timeout configuration optimized")
    print(f"   ✅ Monitoring systems configured")
    print(f"   ✅ Backup and recovery procedures")


def example_troubleshooting():
    """Example: Common troubleshooting scenarios"""
    print("\n🔍 Example 5: Troubleshooting Common Issues")
    print("=" * 50)
    
    scenarios = [
        {
            "issue": "Connection to Ollama fails",
            "symptoms": "Connection timeout, 'Connection refused' errors",
            "solutions": [
                "Check if Ollama service is running: ollama serve",
                "Verify Ollama is listening on correct port (11434)",
                "Check firewall settings",
                "Increase connection timeout in config"
            ]
        },
        {
            "issue": "Requests timing out frequently", 
            "symptoms": "Many timeout events, slow responses",
            "solutions": [
                "Increase request timeout limits",
                "Check network connectivity",
                "Verify AI model is loaded and responsive",
                "Monitor system resources (CPU, memory)"
            ]
        },
        {
            "issue": "Sessions expiring too quickly",
            "symptoms": "Users getting logged out, idle timeouts",
            "solutions": [
                "Increase idle timeout in configuration",
                "Enable session heartbeat",
                "Check for client-side activity updates",
                "Review session duration limits"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🚨 Issue {i}: {scenario['issue']}")
        print(f"   Symptoms: {scenario['symptoms']}")
        print(f"   Solutions:")
        for solution in scenario['solutions']:
            print(f"     • {solution}")


async def example_async_usage():
    """Example: Using StudyForge AI with async/await"""
    print("\n⚡ Example 6: Async/Await Usage")
    print("=" * 50)
    
    print("""
    # Example of using StudyForge AI in async code:
    
    import asyncio
    from main import TimeoutEnhancedAI
    
    async def study_session():
        ai = TimeoutEnhancedAI()
        
        # Ask multiple questions concurrently
        questions = [
            "Explain sorting algorithms",
            "What is machine learning?", 
            "How do databases work?"
        ]
        
        tasks = [ai.query_ollama(q) for q in questions]
        responses = await asyncio.gather(*tasks)
        
        for q, r in zip(questions, responses):
            print(f"Q: {q}")
            print(f"A: {r[:100]}...")
            
        await ai.shutdown()
    
    # Run the async study session
    asyncio.run(study_session())
    """)


def main():
    """Run all examples"""
    print("🎓 StudyForge AI - Usage Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use StudyForge AI")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_timeout_configuration()  
    example_session_management()
    example_production_deployment()
    example_troubleshooting()
    example_async_usage()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("💡 Try running: python src/main.py")
    print("📖 Read more: docs/TIMEOUT_MANAGEMENT_GUIDE.md")
    print("🧪 Run tests: python tests/timeout_functionality_test.py")


if __name__ == "__main__":
    main()