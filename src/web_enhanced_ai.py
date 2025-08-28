#!/usr/bin/env python3
"""
StudyForge AI - Web-Enhanced AI Agent
Combines timeout management with intelligent web search capabilities
"""

import asyncio
import aiohttp
import json
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import our web search modules
from .web_search_engine import WebSearchEngine, SearchResponse, format_search_results
from .search_providers import get_provider, get_providers_for_search_type, AVAILABLE_PROVIDERS
from .content_processor import ContentProcessor, process_web_search, format_processed_results
from .enterprise_timeout_config import EnterpriseTimeoutConfig, TimeoutManager

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
class WebSearchConfig:
    """Configuration for web search functionality"""
    enable_web_search: bool = True
    auto_search_threshold: float = 0.7  # Confidence threshold for auto-search
    max_search_results: int = 10
    max_content_extraction: int = 5
    search_cache_hours: int = 24
    preferred_providers: List[str] = None
    enable_academic_search: bool = True
    enable_news_search: bool = True
    enable_code_search: bool = True
    
    def __post_init__(self):
        if self.preferred_providers is None:
            self.preferred_providers = ['duckduckgo', 'wikipedia']


class WebEnhancedAI:
    """AI Agent with web search capabilities and timeout management"""
    
    def __init__(self):
        # Load configurations
        self.timeout_config = EnterpriseTimeoutConfig(environment="development")
        self.web_config = WebSearchConfig()
        
        # Initialize timeout management
        self.timeout_manager = TimeoutManager(self.timeout_config)
        
        # Initialize web search components
        self.search_engine = None
        self.content_processor = None
        self.http_session = None
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "gpt-oss:20b"
        
        # Session management
        self.session_id = None
        self.conversation_history = []
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'web_searches': 0,
            'successful_searches': 0,
            'cache_hits': 0,
            'processing_time': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"{Fore.GREEN}🌟 StudyForge AI - Web-Enhanced Agent Initialized{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Timeout management: Enterprise-grade{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Web search: {len(AVAILABLE_PROVIDERS)} providers available{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Content processing: Advanced extraction{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Model: {self.model_name}{Style.RESET_ALL}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n{Fore.YELLOW}🛑 Received shutdown signal, cleaning up...{Style.RESET_ALL}")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Create HTTP session with timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.timeout_config.network.total_request_timeout,
                connect=self.timeout_config.network.connection_timeout,
                sock_read=self.timeout_config.network.read_timeout
            )
            
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'StudyForge-AI/1.0 Educational Assistant'}
            )
            
            # Initialize web search engine
            search_config = {
                'max_results_per_provider': 5,
                'total_max_results': self.web_config.max_search_results,
                'request_timeout': self.timeout_config.network.read_timeout,
                'total_timeout': self.timeout_config.network.total_request_timeout,
                'enable_caching': True,
                'cache_max_age_hours': self.web_config.search_cache_hours
            }
            
            self.search_engine = WebSearchEngine(search_config)
            await self.search_engine.create_session()
            
            # Initialize content processor
            processor_config = {
                'max_content_length': 6000,
                'request_timeout': self.timeout_config.network.read_timeout,
                'max_concurrent_extractions': self.web_config.max_content_extraction
            }
            
            self.content_processor = ContentProcessor(processor_config, self.http_session)
            
            # Create session
            self.session_id = f"web_enhanced_session_{int(time.time())}"
            if self.timeout_manager.create_session(self.session_id, "web_enhanced_user"):
                print(f"{Fore.GREEN}✅ Session created: {self.session_id}{Style.RESET_ALL}")
            
            print(f"{Fore.GREEN}🚀 Web-Enhanced AI fully initialized and ready!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Initialization failed: {e}{Style.RESET_ALL}")
            return False
    
    async def query_with_web_enhancement(self, user_query: str, force_web_search: bool = False) -> str:
        """Main query method with web enhancement"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Update session activity
        if self.session_id:
            self.timeout_manager.update_session_activity(self.session_id)
        
        print(f"\n{Fore.LIGHTCYAN_EX}🧠 Processing: {user_query}{Style.RESET_ALL}")
        
        try:
            # Step 1: Determine if web search is needed
            web_context = ""
            if self.web_config.enable_web_search:
                needs_web_search, confidence, reason = await self._analyze_query_for_web_search(user_query)
                
                if force_web_search or needs_web_search:
                    print(f"{Fore.YELLOW}🌐 Web search triggered: {reason} (confidence: {confidence:.2f}){Style.RESET_ALL}")
                    web_context = await self._perform_web_search(user_query)
                    self.stats['web_searches'] += 1
                else:
                    print(f"{Fore.BLUE}💭 Using local knowledge only: {reason}{Style.RESET_ALL}")
            
            # Step 2: Query local AI model with or without web context
            ai_response = await self._query_local_model(user_query, web_context)
            
            # Step 3: Update statistics and history
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_query': user_query,
                'web_search_used': bool(web_context),
                'processing_time': processing_time,
                'response_length': len(ai_response)
            })
            
            return ai_response
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            print(f"{Fore.RED}❌ Query processing failed: {e}{Style.RESET_ALL}")
            
            # Record timeout if applicable
            if self.session_id and "timeout" in str(e).lower():
                self.timeout_manager.record_timeout(self.session_id, "ai_processing", str(e))
            
            return error_msg
    
    async def _analyze_query_for_web_search(self, query: str) -> Tuple[bool, float, str]:
        """Analyze if query needs web search with confidence score"""
        query_lower = query.lower()
        confidence = 0.0
        reasons = []
        
        # High confidence indicators (0.8-1.0)
        high_confidence_patterns = [
            ('latest', 0.9), ('recent', 0.85), ('current', 0.8), ('today', 0.9),
            ('2024', 0.9), ('2023', 0.8), ('this year', 0.9), ('this month', 0.85),
            ('breaking', 0.95), ('news', 0.8), ('trending', 0.85)
        ]
        
        # Medium confidence indicators (0.5-0.7)
        medium_confidence_patterns = [
            ('what happened', 0.7), ('what is', 0.6), ('who is', 0.65),
            ('price of', 0.7), ('cost of', 0.6), ('review of', 0.6),
            ('how to', 0.5), ('when did', 0.7), ('where is', 0.6)
        ]
        
        # Academic/research indicators (0.6-0.8)
        academic_patterns = [
            ('research', 0.7), ('study', 0.6), ('paper', 0.75), ('journal', 0.7),
            ('publication', 0.75), ('findings', 0.7), ('experiment', 0.6)
        ]
        
        # Check patterns
        for pattern, score in high_confidence_patterns:
            if pattern in query_lower:
                confidence = max(confidence, score)
                reasons.append(f"high-priority keyword: '{pattern}'")
        
        for pattern, score in medium_confidence_patterns:
            if pattern in query_lower:
                confidence = max(confidence, score)
                reasons.append(f"search indicator: '{pattern}'")
        
        for pattern, score in academic_patterns:
            if pattern in query_lower:
                confidence = max(confidence, score * 0.9)  # Slightly lower for academic
                reasons.append(f"academic query: '{pattern}'")
        
        # Web-specific requests
        web_indicators = ['website', 'url', 'link', 'online', 'internet']
        for indicator in web_indicators:
            if indicator in query_lower:
                confidence = max(confidence, 0.8)
                reasons.append(f"web-specific request: '{indicator}'")
        
        # Company, product, or person names (heuristic)
        if any(word.istitle() and len(word) > 3 for word in query.split()):
            confidence = max(confidence, 0.6)
            reasons.append("proper nouns detected (entities)")
        
        # Questions about specific topics that might have recent developments
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(query_lower.startswith(q) for q in question_starters):
            if confidence < 0.5:
                confidence = 0.4  # Base question confidence
                reasons.append("factual question")
        
        # Final decision
        needs_search = confidence >= self.web_config.auto_search_threshold
        reason = "; ".join(reasons) if reasons else "no web search triggers"
        
        return needs_search, confidence, reason
    
    async def _perform_web_search(self, query: str) -> str:
        """Perform web search and return context for AI model"""
        try:
            # Determine search type based on query
            search_type = self._determine_search_type(query)
            
            print(f"{Fore.CYAN}🔍 Searching web ({search_type} search)...{Style.RESET_ALL}")
            
            # Perform search
            search_response = await self.search_engine.search(query, search_type, force_web=True)
            
            if not search_response.results:
                return "No relevant web results found."
            
            print(f"{Fore.GREEN}✅ Found {len(search_response.results)} results{Style.RESET_ALL}")
            
            # Process content
            processed_content, ai_context = await process_web_search(search_response, self.http_session)
            
            if processed_content:
                self.stats['successful_searches'] += 1
                print(f"{Fore.GREEN}📚 Processed {len(processed_content)} sources successfully{Style.RESET_ALL}")
                
                # Show brief summary to user
                stats = self.content_processor.get_processing_stats(processed_content)
                print(f"{Fore.MAGENTA}📊 {stats['total_words']} words from {stats['total_sources']} sources{Style.RESET_ALL}")
                
                return ai_context
            else:
                return "Web search completed but content processing failed."
                
        except Exception as e:
            print(f"{Fore.RED}❌ Web search failed: {e}{Style.RESET_ALL}")
            if self.session_id:
                self.timeout_manager.record_timeout(self.session_id, "web_search", str(e))
            return f"Web search encountered an error: {str(e)}"
    
    def _determine_search_type(self, query: str) -> str:
        """Determine the best search type based on query content"""
        query_lower = query.lower()
        
        # Academic/research queries
        if any(term in query_lower for term in ['research', 'study', 'paper', 'journal', 'academic', 'publication']):
            return 'academic'
        
        # News queries
        if any(term in query_lower for term in ['news', 'breaking', 'latest', 'today', 'recent']):
            return 'news'
        
        # Code/programming queries
        if any(term in query_lower for term in ['code', 'programming', 'function', 'library', 'github', 'algorithm']):
            return 'code'
        
        # Discussion/opinion queries
        if any(term in query_lower for term in ['opinion', 'discussion', 'what do people think', 'community']):
            return 'discussion'
        
        return 'general'
    
    async def _query_local_model(self, user_query: str, web_context: str = "") -> str:
        """Query the local Ollama model with optional web context"""
        # Prepare the prompt
        if web_context:
            full_prompt = f"""Web Search Context:
{web_context}

User Question: {user_query}

Please provide a comprehensive answer using both the web search results above and your knowledge. Include relevant citations and links where appropriate."""
        else:
            full_prompt = user_query
        
        # Prepare request data
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
            print(f"{Fore.BLUE}🤖 Querying {self.model_name}...{Style.RESET_ALL}")
            
            # Use timeout-aware HTTP client
            timeout = aiohttp.ClientTimeout(total=self.timeout_config.ai_model.inference_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.ollama_url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get('response', 'No response received from AI model.')
                        
                        print(f"{Fore.GREEN}✅ AI response generated ({len(ai_response)} chars){Style.RESET_ALL}")
                        return ai_response
                    else:
                        error_msg = f"AI model returned status {response.status}"
                        print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
                        return f"I couldn't process your request. {error_msg}"
                        
        except asyncio.TimeoutError:
            timeout_msg = f"AI processing timed out after {self.timeout_config.ai_model.inference_timeout}s"
            print(f"{Fore.RED}⏱️ {timeout_msg}{Style.RESET_ALL}")
            return f"The request took longer than expected to process. {timeout_msg}. Please try a simpler question."
            
        except Exception as e:
            error_msg = f"Error communicating with AI model: {str(e)}"
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def show_session_status(self):
        """Display current session status and statistics"""
        if not self.session_id:
            print(f"{Fore.RED}❌ No active session{Style.RESET_ALL}")
            return
        
        # Get session status from timeout manager
        session_status = self.timeout_manager.get_session_status(self.session_id)
        
        print(f"\n{Fore.CYAN}📊 StudyForge AI - Session Status{Style.RESET_ALL}")
        print("="*50)
        
        if session_status:
            idle_min = session_status['idle_remaining_seconds'] / 60
            session_min = session_status['session_remaining_seconds'] / 60
            
            print(f"Session ID: {self.session_id}")
            print(f"Status: {'🟢 Active' if session_status['is_active'] else '🔴 Expired'}")
            print(f"Idle timeout: {idle_min:.1f} minutes remaining")
            print(f"Session timeout: {session_min:.1f} hours remaining")
            print(f"Timeout count: {session_status['timeout_count']}")
        
        print(f"\n📈 Query Statistics:")
        print(f"Total queries: {self.stats['total_queries']}")
        print(f"Web searches: {self.stats['web_searches']}")
        print(f"Successful searches: {self.stats['successful_searches']}")
        print(f"Average processing time: {self.stats['processing_time']/max(1, self.stats['total_queries']):.2f}s")
        
        if self.search_engine and hasattr(self.search_engine, 'cache'):
            print(f"Cache hits: {self.stats.get('cache_hits', 0)}")
        
        print(f"\n🌐 Web Search Configuration:")
        print(f"Auto-search threshold: {self.web_config.auto_search_threshold}")
        print(f"Max results: {self.web_config.max_search_results}")
        print(f"Preferred providers: {', '.join(self.web_config.preferred_providers)}")
    
    def show_help(self):
        """Display help information"""
        help_text = f"""
{Fore.LIGHTCYAN_EX}🌟 StudyForge AI - Web-Enhanced Assistant{Style.RESET_ALL}

{Fore.LIGHTGREEN_EX}Commands:{Style.RESET_ALL}
  /search <query>     - Force web search for query
  /local <query>      - Use local AI only (no web search)
  /status             - Show session status and statistics
  /providers          - List available search providers
  /help               - Show this help message
  /quit or /exit      - Exit the application

{Fore.LIGHTBLUE_EX}Features:{Style.RESET_ALL}
  🌐 Intelligent Web Search  - Automatically searches when needed
  🤖 Local AI Processing    - Private, local model inference
  ⏱️  Enterprise Timeouts    - Never hangs or freezes
  📚 Content Processing     - Extracts and summarizes web content
  📊 Session Analytics      - Track your usage patterns
  🎯 Smart Caching         - Fast results for repeated queries

{Fore.LIGHTYELLOW_EX}Web Search Triggers:{Style.RESET_ALL}
  • Recent/current events (latest, today, 2024, breaking)
  • Specific people, companies, or products
  • News and trending topics
  • Academic research queries
  • "What happened" or "What is" questions

{Fore.LIGHTMAGENTA_EX}Examples:{Style.RESET_ALL}
  "Latest Python 3.12 features"        → Auto web search
  "What is machine learning?"           → Local knowledge
  "Breaking news about AI today"        → Auto web search
  "/search Python tutorial"             → Forced web search
  "/local explain binary trees"         → Local only

{Fore.WHITE}Your queries are processed intelligently - web search when needed, local knowledge when sufficient.{Style.RESET_ALL}
        """
        print(help_text)
    
    def list_providers(self):
        """List available search providers"""
        print(f"\n{Fore.CYAN}🔍 Available Search Providers:{Style.RESET_ALL}")
        print("="*40)
        
        for name, provider_class in AVAILABLE_PROVIDERS.items():
            status = "✅ Available" if name in self.web_config.preferred_providers else "⚪ Available"
            print(f"{status} {name.title()}")
            if hasattr(provider_class, '__doc__') and provider_class.__doc__:
                doc = provider_class.__doc__.strip().split('\n')[0]
                print(f"    {doc}")
        
        print(f"\n{Fore.YELLOW}💡 Preferred providers: {', '.join(self.web_config.preferred_providers)}{Style.RESET_ALL}")
    
    async def run_interactive_session(self):
        """Run interactive chat session"""
        print(f"\n{Fore.LIGHTGREEN_EX}🎓 Welcome to StudyForge AI - Web-Enhanced Assistant!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type your questions naturally - I'll search the web when needed.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Use /help for commands, /quit to exit.{Style.RESET_ALL}")
        
        while True:
            try:
                # Check session limits
                if self.session_id:
                    session_status = self.timeout_manager.get_session_status(self.session_id)
                    if not session_status or not session_status['is_active']:
                        print(f"{Fore.RED}⚠️ Session expired. Please restart the application.{Style.RESET_ALL}")
                        break
                
                # Get user input
                user_input = input(f"\n{Fore.LIGHTCYAN_EX}You: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    break
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                elif user_input.lower() == '/status':
                    self.show_session_status()
                    continue
                elif user_input.lower() == '/providers':
                    self.list_providers()
                    continue
                elif user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        response = await self.query_with_web_enhancement(query, force_web_search=True)
                    else:
                        print(f"{Fore.RED}❌ Please provide a search query: /search <your query>{Style.RESET_ALL}")
                        continue
                elif user_input.lower().startswith('/local '):
                    query = user_input[7:].strip()
                    if query:
                        response = await self._query_local_model(query)
                    else:
                        print(f"{Fore.RED}❌ Please provide a query: /local <your query>{Style.RESET_ALL}")
                        continue
                else:
                    # Normal query processing
                    response = await self.query_with_web_enhancement(user_input)
                
                # Display response
                print(f"\n{Fore.LIGHTGREEN_EX}StudyForge AI: {Style.RESET_ALL}{response}")
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}⚠️ Interrupted by user{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}⚠️ Input stream closed{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}👋 Thank you for using StudyForge AI! Keep learning! 📚✨{Style.RESET_ALL}")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        print(f"{Fore.CYAN}🛑 Shutting down StudyForge AI...{Style.RESET_ALL}")
        
        try:
            # Close web search engine
            if self.search_engine:
                await self.search_engine.close()
            
            # Close HTTP session
            if self.http_session:
                await self.http_session.close()
            
            # Shutdown timeout manager
            if self.timeout_manager:
                self.timeout_manager.shutdown()
            
            # Display final statistics
            print(f"{Fore.GREEN}📊 Session Summary:{Style.RESET_ALL}")
            print(f"  • Total queries: {self.stats['total_queries']}")
            print(f"  • Web searches: {self.stats['web_searches']}")
            print(f"  • Total processing time: {self.stats['processing_time']:.2f}s")
            
            print(f"{Fore.GREEN}✅ Shutdown complete. Goodbye!{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error during shutdown: {e}{Style.RESET_ALL}")


async def main():
    """Main entry point for Web-Enhanced StudyForge AI"""
    ai = WebEnhancedAI()
    
    try:
        # Initialize the system
        if not await ai.initialize():
            print(f"{Fore.RED}❌ Failed to initialize StudyForge AI{Style.RESET_ALL}")
            return 1
        
        # Run interactive session
        await ai.run_interactive_session()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Received interrupt signal{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}💥 Fatal error: {e}{Style.RESET_ALL}")
        return 1
    finally:
        await ai.shutdown()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        exit(0)