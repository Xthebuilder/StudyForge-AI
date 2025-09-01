#!/usr/bin/env python3
"""
Test script for StudyForge AI database integration
Tests database functionality, memory management, and rolling memory system
"""

import asyncio
import time
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database_manager import DatabaseManager, MemoryConfig, ConversationEntry
    from src.memory_compressor import MemoryCompressor
    from src.web_search_engine import WebSearchEngine
    from src.web_enhanced_ai import WebEnhancedAI
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the StudyForge-AI directory")
    sys.exit(1)

class DatabaseIntegrationTest:
    def __init__(self):
        self.db_manager = DatabaseManager(db_path="test_studyforge.db")
        self.memory_compressor = MemoryCompressor()
        self.test_user = "test_user_001"
        
    def run_all_tests(self):
        """Run comprehensive database integration tests"""
        print("🧪 StudyForge AI - Database Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("Database Initialization", self.test_database_init),
            ("Conversation Storage", self.test_conversation_storage),
            ("Memory Management", self.test_memory_management),
            ("Search Caching", self.test_search_caching),
            ("Rolling Memory", self.test_rolling_memory),
            ("Memory Compression", self.test_memory_compression),
            ("Contextual Memory Retrieval", self.test_contextual_memory),
            ("User Statistics", self.test_user_statistics)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                if result:
                    print(f"✅ PASS ({duration:.2f}s)")
                    results[test_name] = "PASS"
                else:
                    print(f"❌ FAIL ({duration:.2f}s)")
                    results[test_name] = "FAIL"
                    
            except Exception as e:
                print(f"💥 ERROR: {e}")
                results[test_name] = "ERROR"
        
        # Summary
        print(f"\n📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for r in results.values() if r == "PASS")
        total = len(results)
        
        for test_name, result in results.items():
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[result]
            print(f"{status_emoji} {test_name}: {result}")
        
        print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Database integration is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        # Cleanup
        try:
            os.remove("test_studyforge.db")
            print("\n🧹 Test database cleaned up")
        except:
            pass
        
        return passed == total
    
    def test_database_init(self):
        """Test database initialization"""
        try:
            # Test table creation
            with self.db_manager.get_connection() as conn:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                table_names = [row['name'] for row in tables]
                
                expected_tables = ['conversations', 'search_cache', 'user_profiles', 'memory_summaries']
                
                for table in expected_tables:
                    if table not in table_names:
                        print(f"❌ Missing table: {table}")
                        return False
                
                print(f"✅ All required tables created: {table_names}")
                return True
                
        except Exception as e:
            print(f"❌ Database init failed: {e}")
            return False
    
    def test_conversation_storage(self):
        """Test conversation storage and retrieval"""
        try:
            # Add test conversations
            conv_id1 = self.db_manager.add_conversation(
                user_id=self.test_user,
                message_type='user',
                content='What is machine learning?',
                search_context={'test': True}
            )
            
            conv_id2 = self.db_manager.add_conversation(
                user_id=self.test_user,
                message_type='ai',
                content='Machine learning is a subset of artificial intelligence...',
            )
            
            print(f"✅ Stored conversations with IDs: {conv_id1}, {conv_id2}")
            
            # Retrieve conversations
            history = self.db_manager.get_conversation_history(self.test_user, limit=10)
            
            if len(history) >= 2:
                print(f"✅ Retrieved {len(history)} conversations")
                print(f"   Latest: {history[0].content[:50]}...")
                return True
            else:
                print(f"❌ Expected at least 2 conversations, got {len(history)}")
                return False
                
        except Exception as e:
            print(f"❌ Conversation storage failed: {e}")
            return False
    
    def test_memory_management(self):
        """Test memory management features"""
        try:
            # Add many conversations to trigger memory management
            for i in range(15):
                self.db_manager.add_conversation(
                    user_id=self.test_user,
                    message_type='user',
                    content=f'Test question {i+1}',
                )
                
                self.db_manager.add_conversation(
                    user_id=self.test_user,
                    message_type='ai',
                    content=f'Test response {i+1}. This is a longer response to test the memory management system.',
                )
            
            # Check if memory management triggered
            stats = self.db_manager.get_user_stats(self.test_user)
            print(f"✅ Total conversations: {stats['total_conversations']}")
            print(f"✅ Memory summaries: {stats['total_summaries']}")
            
            return stats['total_conversations'] > 0
            
        except Exception as e:
            print(f"❌ Memory management failed: {e}")
            return False
    
    def test_search_caching(self):
        """Test search result caching"""
        try:
            # Cache some search results
            test_results = [
                {'title': 'Test Result 1', 'url': 'http://test1.com', 'snippet': 'Test snippet 1'},
                {'title': 'Test Result 2', 'url': 'http://test2.com', 'snippet': 'Test snippet 2'}
            ]
            
            self.db_manager.cache_search_result('test query', 'test_provider', test_results)
            print("✅ Cached search results")
            
            # Retrieve cached results
            cached = self.db_manager.get_cached_search('test query', 'test_provider')
            
            if cached and len(cached) == 2:
                print(f"✅ Retrieved {len(cached)} cached results")
                return True
            else:
                print(f"❌ Cache retrieval failed")
                return False
                
        except Exception as e:
            print(f"❌ Search caching failed: {e}")
            return False
    
    def test_rolling_memory(self):
        """Test rolling memory functionality"""
        try:
            # Create a database manager with small limits for testing
            test_config = MemoryConfig(
                max_conversations=20,
                rolling_window_size=10,
                importance_threshold=0.3
            )
            
            test_db = DatabaseManager(db_path="test_rolling.db", config=test_config)
            
            # Add conversations beyond the limit
            for i in range(25):
                test_db.add_conversation(
                    user_id="rolling_test_user",
                    message_type='user' if i % 2 == 0 else 'ai',
                    content=f'Rolling test message {i+1}',
                )
            
            # Check if rolling occurred
            stats = test_db.get_user_stats("rolling_test_user")
            history = test_db.get_conversation_history("rolling_test_user")
            
            print(f"✅ Final conversation count: {stats['total_conversations']}")
            print(f"✅ Memory summaries created: {stats['total_summaries']}")
            
            # Cleanup
            os.remove("test_rolling.db")
            
            return stats['total_conversations'] <= 20
            
        except Exception as e:
            print(f"❌ Rolling memory failed: {e}")
            return False
    
    def test_memory_compression(self):
        """Test memory compression system"""
        try:
            # Test conversation compression
            test_conversations = [
                {'content': 'What is Python programming?', 'message_type': 'user', 'timestamp': time.time(), 'importance_score': 0.8},
                {'content': 'Python is a high-level programming language...', 'message_type': 'ai', 'timestamp': time.time(), 'importance_score': 0.7},
                {'content': 'How do I install Python?', 'message_type': 'user', 'timestamp': time.time(), 'importance_score': 0.6}
            ]
            
            compressed = self.memory_compressor.compress_conversation_batch(test_conversations)
            
            print(f"✅ Compressed {len(test_conversations)} conversations")
            print(f"   Summary: {compressed['summary'][:100]}...")
            print(f"   Key topics: {compressed['key_topics']}")
            print(f"   Compression ratio: {compressed['compression_metadata']['compression_ratio']:.2f}")
            
            return len(compressed['summary']) > 0 and len(compressed['key_topics']) > 0
            
        except Exception as e:
            print(f"❌ Memory compression failed: {e}")
            return False
    
    def test_contextual_memory(self):
        """Test contextual memory retrieval"""
        try:
            # Get contextual memory
            context = self.db_manager.get_contextual_memory(self.test_user, 'machine learning')
            
            print(f"✅ Retrieved contextual memory")
            print(f"   Recent conversations: {len(context['recent_conversations'])}")
            print(f"   Important conversations: {len(context['important_conversations'])}")
            print(f"   Historical summaries: {len(context['historical_summaries'])}")
            
            return 'recent_conversations' in context
            
        except Exception as e:
            print(f"❌ Contextual memory failed: {e}")
            return False
    
    def test_user_statistics(self):
        """Test user statistics retrieval"""
        try:
            stats = self.db_manager.get_user_stats(self.test_user)
            
            print(f"✅ User statistics:")
            print(f"   Total conversations: {stats['total_conversations']}")
            print(f"   Average importance: {stats['avg_importance']:.2f}")
            print(f"   Total summaries: {stats['total_summaries']}")
            
            return isinstance(stats['total_conversations'], int)
            
        except Exception as e:
            print(f"❌ User statistics failed: {e}")
            return False

async def test_web_enhanced_ai():
    """Test the complete web-enhanced AI with database"""
    print("\n🤖 Testing Web-Enhanced AI with Database Integration")
    print("-" * 60)
    
    try:
        # Create AI instance
        ai = WebEnhancedAI(user_id="test_integration_user")
        
        # Initialize
        if not await ai.initialize():
            print("❌ AI initialization failed")
            return False
        
        print("✅ AI initialized with database support")
        
        # Test a simple query (should use memory)
        response = await ai.query_with_web_enhancement("What is 2+2?", force_web_search=False)
        print(f"✅ Local query response: {response[:100]}...")
        
        # Test memory retrieval
        stats = ai.db_manager.get_user_stats(ai.user_id)
        print(f"✅ User has {stats['total_conversations']} conversations stored")
        
        # Cleanup
        await ai.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Web-Enhanced AI test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 StudyForge AI - Complete Database Integration Test")
    print("=" * 80)
    
    # Database tests
    test_suite = DatabaseIntegrationTest()
    db_success = test_suite.run_all_tests()
    
    # AI integration test
    ai_success = asyncio.run(test_web_enhanced_ai())
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    print(f"🗄️ Database Integration: {'✅ PASS' if db_success else '❌ FAIL'}")
    print(f"🤖 AI Integration: {'✅ PASS' if ai_success else '❌ FAIL'}")
    
    if db_success and ai_success:
        print("\n🎉 ALL SYSTEMS GO! StudyForge AI with Database is ready!")
        print("💡 You can now run: python agent.py")
    else:
        print("\n⚠️ Some tests failed. Please check the output above.")
        print("💡 Fix issues before running the main agent")
    
    return db_success and ai_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)