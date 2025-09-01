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
from .database_manager import DatabaseManager, ConversationEntry, MemoryConfig
from .memory_compressor import MemoryCompressor
from .enterprise_logger import EnterpriseLogger, SecurityEventType, log_performance

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
    
    def __init__(self, user_id: str = None, rag_processor=None):
        # Initialize enterprise logging first
        self.logger = EnterpriseLogger()
        self.user_id = user_id or f"user_{int(time.time())}"
        
        # Log system initialization
        self.logger.audit('system_initialization', 'web_enhanced_ai', user_id=self.user_id)
        
        # Load configurations
        self.timeout_config = EnterpriseTimeoutConfig(environment="development")
        self.web_config = WebSearchConfig()
        
        # Initialize timeout management
        self.timeout_manager = TimeoutManager(self.timeout_config)
        
        # Initialize web search components
        self.search_engine = None
        self.content_processor = None
        self.http_session = None
        
        # Database and memory management
        memory_config = MemoryConfig()
        self.db_manager = DatabaseManager(config=memory_config)
        self.memory_compressor = MemoryCompressor()
        
        # RAG integration
        self.rag_processor = rag_processor
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "deepseek-r1:14b"  # Smaller, faster model
        
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
        
        self.logger.info('StudyForge AI Web-Enhanced Agent Initialized', 
                        user_id=self.user_id,
                        timeout_management='enterprise-grade',
                        web_providers=len(AVAILABLE_PROVIDERS),
                        model=self.model_name)
        
        print(f"{Fore.GREEN}🌟 StudyForge AI - Web-Enhanced Agent Initialized{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Timeout management: Enterprise-grade{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Web search: {len(AVAILABLE_PROVIDERS)} providers available{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Content processing: Advanced extraction{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Model: {self.model_name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Enterprise logging: Enabled{Style.RESET_ALL}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.audit('system_shutdown', 'signal_handler', outcome='initiated', signal=signum)
        print(f"\n{Fore.YELLOW}🛑 Received shutdown signal, cleaning up...{Style.RESET_ALL}")
        asyncio.create_task(self.shutdown())
    
    @log_performance('system_initialization')
    async def initialize(self) -> bool:
        """Initialize all components"""
        request_id = f"init_{int(time.time())}"
        
        with self.logger.request_context(request_id, user_id=self.user_id):
            self.logger.info('Starting system initialization')
            
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
                
                self.search_engine = WebSearchEngine(search_config, db_manager=self.db_manager)
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
                    self.logger.audit('session_created', 'timeout_manager', session_id=self.session_id)
                    print(f"{Fore.GREEN}✅ Session created: {self.session_id}{Style.RESET_ALL}")
            
                self.logger.info('System initialization completed successfully')
                print(f"{Fore.GREEN}🚀 Web-Enhanced AI fully initialized and ready!{Style.RESET_ALL}")
                return True
                
            except Exception as e:
                self.logger.critical('System initialization failed', error=e, user_id=self.user_id)
                self.logger.security(SecurityEventType.SYSTEM_BREACH, 
                                    'Critical system initialization failure', 
                                    severity='high', error=str(e))
                print(f"{Fore.RED}❌ Initialization failed: {e}{Style.RESET_ALL}")
                return False
    
    @log_performance('query_processing')
    async def query_with_web_enhancement(self, user_query: str, force_web_search: bool = False) -> str:
        """Main query method with web enhancement"""
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        self.stats['total_queries'] += 1
        
        with self.logger.request_context(query_id, user_id=self.user_id, session_id=self.session_id):
            self.logger.audit('query_received', 'web_enhanced_ai', 
                            query_length=len(user_query),
                            force_web_search=force_web_search)
        
            # Update session activity
            if self.session_id:
                self.timeout_manager.update_session_activity(self.session_id)
            
            print(f"\n{Fore.LIGHTCYAN_EX}🧠 Processing: {user_query}{Style.RESET_ALL}")
        
            try:
                # Step 1: Store user query in database
                search_context = None
                self.db_manager.add_conversation(
                    user_id=self.user_id,
                    message_type='user',
                    content=user_query,
                    search_context=search_context
                )
                
                # Step 2: Get conversation context for AI
                contextual_memory = self.db_manager.get_contextual_memory(self.user_id, user_query)
                
                # Step 3: Determine if web search is needed
                web_context = ""
                if self.web_config.enable_web_search:
                    needs_web_search, confidence, reason = await self._analyze_query_for_web_search(user_query)
                    
                    if force_web_search or needs_web_search:
                        self.logger.audit('web_search_triggered', 'query_analysis', 
                                         confidence=confidence, reason=reason, forced=force_web_search)
                        print(f"{Fore.YELLOW}🌐 Web search triggered: {reason} (confidence: {confidence:.2f}){Style.RESET_ALL}")
                    web_context = await self._perform_web_search(user_query)
                    self.stats['web_searches'] += 1
                    
                    # Update search context
                    search_context = {
                        'web_search_used': True,
                        'search_confidence': confidence,
                        'search_reason': reason
                    }
                else:
                    self.logger.debug('Using local knowledge only', reason=reason, confidence=confidence)
                    print(f"{Fore.BLUE}💭 Using local knowledge only: {reason}{Style.RESET_ALL}")
            
                # Step 4: Query local AI model with context
                ai_response = await self._query_local_model(user_query, web_context, contextual_memory)
                
                # Step 5: Store AI response in database
                self.db_manager.add_conversation(
                    user_id=self.user_id,
                    message_type='ai',
                    content=ai_response,
                    search_context=search_context
                )
            
                # Step 6: Update statistics and history
                processing_time = time.time() - start_time
                self.stats['processing_time'] += processing_time
                
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_query': user_query,
                    'web_search_used': bool(web_context),
                    'processing_time': processing_time,
                    'response_length': len(ai_response)
                })
                
                self.logger.audit('query_completed', 'web_enhanced_ai',
                                outcome='success',
                                processing_time=processing_time,
                                response_length=len(ai_response),
                                web_search_used=bool(web_context))
                
                return ai_response
            
            except Exception as e:
                error_msg = f"I encountered an error while processing your request: {str(e)}"
                
                self.logger.error('Query processing failed', error=e, 
                                query_length=len(user_query),
                                processing_time=time.time() - start_time)
                
                # Check for security-relevant errors
                if any(term in str(e).lower() for term in ['permission', 'access', 'unauthorized', 'forbidden']):
                    self.logger.security(SecurityEventType.AUTHORIZATION, 
                                       f'Query processing authorization error: {str(e)}',
                                       severity='medium')
                
                print(f"{Fore.RED}❌ Query processing failed: {e}{Style.RESET_ALL}")
                
                # Record timeout if applicable
                if self.session_id and "timeout" in str(e).lower():
                    self.timeout_manager.record_timeout(self.session_id, "ai_processing", str(e))
                    self.logger.audit('timeout_recorded', 'timeout_manager', 
                                    session_id=self.session_id, error=str(e))
                
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
        
        # Capability inquiry patterns - questions about what the AI can do
        capability_patterns = [
            ('can you search', 0.9), ('do you search', 0.8), ('are you able to search', 0.9),
            ('search the web', 0.95), ('web search', 0.9), ('google search', 0.85),
            ('find information', 0.7), ('look up', 0.6), ('search for', 0.8)
        ]
        
        for pattern, score in capability_patterns:
            if pattern in query_lower:
                confidence = max(confidence, score)
                reasons.append(f"capability inquiry: '{pattern}'")
        
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
    
    @log_performance('web_search')
    async def _perform_web_search(self, query: str) -> str:
        """Perform web search and return context for AI model"""
        search_id = f"search_{int(time.time() * 1000)}"
        
        with self.logger.request_context(search_id, user_id=self.user_id):
            self.logger.info('Starting web search', query_length=len(query))
            
            try:
                # Determine search type based on query
                search_type = self._determine_search_type(query)
            
                self.logger.audit('web_search_initiated', 'search_engine', 
                                search_type=search_type, query_length=len(query))
                print(f"{Fore.CYAN}🔍 Searching web ({search_type} search)...{Style.RESET_ALL}")
                
                # Perform search
                search_response = await self.search_engine.search(query, search_type, force_web=True)
            
                if not search_response.results:
                    self.logger.warning('Web search returned no results', search_type=search_type)
                    return "No relevant web results found."
            
                self.logger.info('Web search completed', 
                                results_count=len(search_response.results),
                                search_type=search_type)
                print(f"{Fore.GREEN}✅ Found {len(search_response.results)} results{Style.RESET_ALL}")
            
                # Process content
                processed_content, ai_context = await process_web_search(search_response, self.http_session)
                
                if processed_content:
                    self.stats['successful_searches'] += 1
                    stats = self.content_processor.get_processing_stats(processed_content)
                    
                    self.logger.audit('content_processing_completed', 'content_processor',
                                    outcome='success',
                                    sources_processed=len(processed_content),
                                    total_words=stats['total_words'],
                                    total_sources=stats['total_sources'])
                    
                    print(f"{Fore.GREEN}📚 Processed {len(processed_content)} sources successfully{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}📊 {stats['total_words']} words from {stats['total_sources']} sources{Style.RESET_ALL}")
                    
                    return ai_context
                else:
                    self.logger.error('Content processing failed despite successful search')
                    return "Web search completed but content processing failed."
                
            except Exception as e:
                self.logger.error('Web search failed', error=e, search_type=search_type)
                
                # Check for rate limiting or blocking
                if any(term in str(e).lower() for term in ['rate limit', 'blocked', 'forbidden', '429']):
                    self.logger.security(SecurityEventType.RATE_LIMIT, 
                                       f'Web search rate limited or blocked: {str(e)}',
                                       severity='medium')
                
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
    
    @log_performance('ai_model_query')
    async def _query_local_model(self, user_query: str, web_context: str = "", contextual_memory: Dict[str, Any] = None) -> str:
        """Query the local Ollama model with optional web context, conversation memory, and RAG context"""
        model_query_id = f"model_{int(time.time() * 1000)}"
        
        with self.logger.request_context(model_query_id, user_id=self.user_id):
            self.logger.info('Starting AI model query', 
                           query_length=len(user_query),
                           has_web_context=bool(web_context),
                           has_memory_context=bool(contextual_memory))
            
            # Get RAG context if available
            rag_context = ""
            if self.rag_processor:
                try:
                    relevant_chunks, rag_context_str = self.rag_processor.retrieve_context(user_query)
                    if rag_context_str:
                        rag_context = f"Document Context:\n{rag_context_str}\n\n"
                        self.logger.audit('rag_context_retrieved', 'rag_processor',
                                        chunks_count=len(relevant_chunks),
                                        context_length=len(rag_context_str))
                        print(f"{Fore.BLUE}📄 Retrieved {len(relevant_chunks)} relevant document chunks{Style.RESET_ALL}")
                except Exception as e:
                    self.logger.error('RAG context retrieval failed', error=e)
                    print(f"{Fore.YELLOW}⚠️ RAG context retrieval failed: {e}{Style.RESET_ALL}")
            
            # Prepare the prompt with memory context
            memory_context = ""
            if contextual_memory:
                # Recent conversations
                if contextual_memory.get('recent_conversations'):
                    recent_msgs = []
                    for conv in contextual_memory['recent_conversations'][:5]:  # Last 5 messages
                        recent_msgs.append(f"{conv.message_type}: {conv.content[:100]}...")
                    memory_context += f"Recent conversation:\n" + "\n".join(recent_msgs) + "\n\n"
                
                # Historical summaries
                if contextual_memory.get('historical_summaries'):
                    summaries = []
                    for summary in contextual_memory['historical_summaries'][:3]:
                        summaries.append(f"- {summary['summary']}")
                    memory_context += f"Previous topics discussed:\n" + "\n".join(summaries) + "\n\n"
            
            # Build full prompt
            prompt_parts = []
            
            if memory_context:
                prompt_parts.append(f"Conversation Context:\n{memory_context}")
            
            if rag_context:
                prompt_parts.append(rag_context)
            
            if web_context:
                prompt_parts.append(f"Web Search Results:\n{web_context}")
            
            prompt_parts.append(f"Current Question: {user_query}")
            
            if web_context or memory_context or rag_context:
                context_sources = []
                if memory_context:
                    context_sources.append("previous conversations")
                if rag_context:
                    context_sources.append("uploaded documents")
                if web_context:
                    context_sources.append("web search results")
                
                sources_text = ", ".join(context_sources)
                prompt_parts.append(f"Please provide a comprehensive answer using the context from {sources_text}. Reference specific sources when relevant.")
            
            full_prompt = "\n".join(prompt_parts)
            
            # Prepare request data with streaming enabled
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": True,  # Enable streaming for faster responses
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
        
            try:
                # Get the configured 5-minute timeout
                ai_timeout = self.timeout_config.ai_model.inference_timeout  # 300 seconds
                self.logger.audit('ai_model_query_initiated', 'ollama_client',
                                model=self.model_name,
                                prompt_length=len(full_prompt),
                                timeout=ai_timeout)
                
                print(f"{Fore.BLUE}🤖 Querying {self.model_name}...{Style.RESET_ALL}")
                
                # Use existing HTTP session with 5-minute timeout override
                timeout = aiohttp.ClientTimeout(total=ai_timeout)
                
                # Handle streaming response
                async with self.http_session.post(self.ollama_url, json=data, timeout=timeout) as response:
                    if response.status == 200:
                        ai_response = ""
                        chunk_count = 0
                        
                        # Process streaming chunks
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                try:
                                    chunk_text = chunk.decode('utf-8')
                                    # Handle multiple JSON objects in chunk
                                    for line in chunk_text.strip().split('\n'):
                                        if line.strip():
                                            try:
                                                chunk_data = json.loads(line)
                                                if 'response' in chunk_data:
                                                    ai_response += chunk_data['response']
                                                    chunk_count += 1
                                                    
                                                    # Show progress every 10 chunks
                                                    if chunk_count % 10 == 0:
                                                        print(f"{Fore.BLUE}📡 Streaming... ({len(ai_response)} chars){Style.RESET_ALL}")
                                                
                                                # Check if done
                                                if chunk_data.get('done', False):
                                                    break
                                            except json.JSONDecodeError:
                                                continue  # Skip malformed JSON
                                except UnicodeDecodeError:
                                    continue  # Skip malformed chunks
                        
                        if not ai_response:
                            ai_response = 'No response received from AI model.'
                        
                        self.logger.audit('ai_model_response_received', 'ollama_client',
                                        outcome='success',
                                        response_length=len(ai_response),
                                        model=self.model_name,
                                        chunks_processed=chunk_count)
                        
                        print(f"{Fore.GREEN}✅ AI streaming response completed ({len(ai_response)} chars, {chunk_count} chunks){Style.RESET_ALL}")
                        return ai_response
                    else:
                        error_msg = f"AI model returned status {response.status}"
                        self.logger.error('AI model HTTP error', status_code=response.status, model=self.model_name)
                        print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
                        return f"I couldn't process your request. {error_msg}"
                        
            except asyncio.TimeoutError:
                timeout_msg = f"AI processing timed out after {self.timeout_config.ai_model.inference_timeout}s"
                
                self.logger.error('AI model timeout', 
                                timeout_seconds=self.timeout_config.ai_model.inference_timeout,
                                model=self.model_name)
                
                self.logger.security(SecurityEventType.SUSPICIOUS_ACTIVITY,
                                   'AI model timeout - potential resource exhaustion',
                                   severity='low')
                
                print(f"{Fore.RED}⏱️ {timeout_msg}{Style.RESET_ALL}")
                return f"The request took longer than expected to process. {timeout_msg}. Please try a simpler question."
    
    async def query_with_streaming(self, user_query: str, websocket=None, force_web_search: bool = False):
        """Query with real-time streaming to WebSocket"""
        start_time = time.time()
        query_id = f"stream_{int(start_time * 1000)}"
        self.stats['total_queries'] += 1
        
        with self.logger.request_context(query_id, user_id=self.user_id, session_id=self.session_id):
            self.logger.audit('streaming_query_received', 'web_enhanced_ai', 
                            query_length=len(user_query),
                            force_web_search=force_web_search)
            
            try:
                # Update session activity
                if self.session_id:
                    self.timeout_manager.update_session_activity(self.session_id)
                
                # Step 1: Web search if needed
                search_context = None
                if force_web_search or self.should_trigger_web_search(user_query):
                    if websocket:
                        await websocket.send_json({
                            "type": "status", 
                            "message": "🔍 Searching the web..."
                        })
                    
                    search_context = await self._perform_web_search(user_query)
                
                # Step 2: Prepare prompt
                memory_context = None
                if self.db_manager:
                    memory_context = self.db_manager.get_contextual_memory(self.user_id, user_query, max_recent=10)
                
                # Get RAG context
                rag_context = ""
                if self.rag_processor:
                    try:
                        relevant_chunks, rag_context_str = self.rag_processor.retrieve_context(user_query)
                        if rag_context_str:
                            rag_context = f"Document Context:\n{rag_context_str}\n\n"
                    except Exception as e:
                        self.logger.error('RAG context retrieval failed', error=e)
                
                # Build prompt
                prompt_parts = []
                if memory_context and memory_context.get('recent_conversations'):
                    recent_msgs = []
                    for conv in memory_context['recent_conversations'][:3]:
                        recent_msgs.append(f"{conv.message_type}: {conv.content[:100]}...")
                    prompt_parts.append(f"Recent conversation:\n" + "\n".join(recent_msgs) + "\n\n")
                
                if rag_context:
                    prompt_parts.append(rag_context)
                
                if search_context:
                    prompt_parts.append(search_context)
                    prompt_parts.append(f"Please provide a comprehensive answer using the context above. Reference specific sources when relevant.")
                
                prompt_parts.append(f"User: {user_query}")
                full_prompt = "\n".join(prompt_parts)
                
                # Step 3: Stream AI response
                if websocket:
                    await websocket.send_json({
                        "type": "status",
                        "message": "🤖 Generating response..."
                    })
                
                return await self._stream_ai_response(full_prompt, websocket)
                
            except Exception as e:
                error_msg = f"Streaming error: {str(e)}"
                self.logger.error('Streaming query error', error=e)
                if websocket:
                    await websocket.send_json({
                        "type": "error",
                        "error": error_msg
                    })
                return error_msg
    
    async def _stream_ai_response(self, prompt: str, websocket=None):
        """Stream AI model response in real-time"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        ai_timeout = self.timeout_config.ai_model.inference_timeout
        timeout = aiohttp.ClientTimeout(total=ai_timeout)
        
        try:
            async with self.http_session.post(self.ollama_url, json=data, timeout=timeout) as response:
                if response.status == 200:
                    ai_response = ""
                    chunk_count = 0
                    
                    async for chunk in response.content.iter_chunked(1024):  # Smaller chunks for faster streaming
                        if chunk:
                            try:
                                chunk_text = chunk.decode('utf-8')
                                for line in chunk_text.strip().split('\n'):
                                    if line.strip():
                                        try:
                                            chunk_data = json.loads(line)
                                            if 'response' in chunk_data and chunk_data['response']:
                                                chunk_text = chunk_data['response']
                                                ai_response += chunk_text
                                                chunk_count += 1
                                                
                                                # Send streaming chunk to WebSocket
                                                if websocket:
                                                    await websocket.send_json({
                                                        "type": "chunk",
                                                        "chunk": chunk_text,
                                                        "total_length": len(ai_response)
                                                    })
                                            
                                            if chunk_data.get('done', False):
                                                break
                                        except json.JSONDecodeError:
                                            continue
                            except UnicodeDecodeError:
                                continue
                    
                    if websocket:
                        await websocket.send_json({
                            "type": "complete",
                            "response": ai_response,
                            "chunks_processed": chunk_count
                        })
                    
                    return ai_response
                else:
                    error_msg = f"AI model returned status {response.status}"
                    if websocket:
                        await websocket.send_json({
                            "type": "error",
                            "error": error_msg
                        })
                    return error_msg
                    
        except asyncio.TimeoutError:
            timeout_msg = "AI response timed out"
            if websocket:
                await websocket.send_json({
                    "type": "error", 
                    "error": timeout_msg
                })
            return timeout_msg
                
        except Exception as e:
            error_msg = f"Error communicating with AI model: {str(e)}"
            
            self.logger.error('AI model communication error', error=e, model=self.model_name)
            
            # Check for connection issues that might indicate security problems
            if 'connection' in str(e).lower() or 'network' in str(e).lower():
                self.logger.security(SecurityEventType.SUSPICIOUS_ACTIVITY,
                                   f'AI model connection error: {str(e)}',
                                   severity='medium')
            
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
        
        # Memory statistics
        print(f"\n🧠 Memory Statistics:")
        user_stats = self.db_manager.get_user_stats(self.user_id)
        print(f"Total conversations: {user_stats['total_conversations']}")
        print(f"Memory summaries: {user_stats['total_summaries']}")
        print(f"Summarized conversations: {user_stats['summarized_conversations']}")
        print(f"Average importance: {user_stats['avg_importance']:.2f}")
        
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
                        response = await self._query_local_model(query, "", None)
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
        self.logger.audit('system_shutdown', 'web_enhanced_ai', outcome='initiated')
        print(f"{Fore.CYAN}🛑 Shutting down StudyForge AI...{Style.RESET_ALL}")
        
        try:
            # Close web search engine
            if self.search_engine:
                await self.search_engine.close()
                self.logger.info('Web search engine closed')
            
            # Close HTTP session
            if self.http_session:
                await self.http_session.close()
                self.logger.info('HTTP session closed')
            
            # Shutdown timeout manager
            if self.timeout_manager:
                self.timeout_manager.shutdown()
                self.logger.info('Timeout manager shutdown')
            
            # Log final statistics
            self.logger.audit('session_summary', 'web_enhanced_ai',
                            outcome='success',
                            total_queries=self.stats['total_queries'],
                            web_searches=self.stats['web_searches'],
                            total_processing_time=self.stats['processing_time'])
            
            # Display final statistics
            print(f"{Fore.GREEN}📊 Session Summary:{Style.RESET_ALL}")
            print(f"  • Total queries: {self.stats['total_queries']}")
            print(f"  • Web searches: {self.stats['web_searches']}")
            print(f"  • Total processing time: {self.stats['processing_time']:.2f}s")
            
            self.logger.audit('system_shutdown', 'web_enhanced_ai', outcome='completed')
            print(f"{Fore.GREEN}✅ Shutdown complete. Goodbye!{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.critical('Shutdown error', error=e)
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
        if 'ai' in locals():
            ai.logger.critical('Fatal system error', error=e)
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