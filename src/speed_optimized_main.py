#!/usr/bin/env python3
"""
StudyForge AI - Speed-Optimized Main Entry Point
Implements all performance optimizations identified in testing
"""

import asyncio
import aiohttp
import json
import time
import signal
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import hashlib
import weakref
from concurrent.futures import ThreadPoolExecutor

# Pre-compile regex patterns for speed (identified optimization)
SEARCH_TRIGGER_PATTERNS = {
    'high_confidence': re.compile(r'\b(latest|recent|current|today|2024|2023|breaking|news|trending)\b', re.IGNORECASE),
    'medium_confidence': re.compile(r'\b(what happened|what is|who is|price of|cost of|review of|how to|when did|where is)\b', re.IGNORECASE),
    'academic': re.compile(r'\b(research|study|paper|journal|publication|findings|experiment)\b', re.IGNORECASE),
    'web_specific': re.compile(r'\b(website|url|link|online|internet)\b', re.IGNORECASE)
}

# DNS cache for speed optimization
DNS_CACHE: Dict[str, str] = {}

# Request deduplication cache
REQUEST_CACHE: Dict[str, Tuple[Any, float]] = {}
REQUEST_CACHE_TTL = 300  # 5 minutes

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


class ConnectionPool:
    """Optimized connection pool for HTTP requests"""
    
    def __init__(self, max_connections: int = 100, max_per_host: int = 10):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_per_host,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=30,
            use_dns_cache=True
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create optimized HTTP session"""
        if self.session is None or self.session.closed:
            async with self._lock:
                if self.session is None or self.session.closed:
                    timeout = aiohttp.ClientTimeout(
                        total=120,
                        connect=10,
                        sock_read=60
                    )
                    
                    self.session = aiohttp.ClientSession(
                        connector=self.connector,
                        timeout=timeout,
                        headers={
                            'User-Agent': 'StudyForge-AI/1.0 Speed-Optimized',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive'
                        }
                    )
        
        return self.session
    
    async def close(self):
        """Close connection pool"""
        if self.session and not self.session.closed:
            await self.session.close()
        if not self.connector.closed:
            await self.connector.close()


class RequestDeduplicator:
    """Prevents duplicate requests and caches responses"""
    
    @staticmethod
    def get_request_key(url: str, params: Dict = None, data: Dict = None) -> str:
        """Generate unique key for request deduplication"""
        key_parts = [url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    @staticmethod
    def get_cached_response(request_key: str) -> Optional[Any]:
        """Get cached response if still valid"""
        if request_key in REQUEST_CACHE:
            response, timestamp = REQUEST_CACHE[request_key]
            if time.time() - timestamp < REQUEST_CACHE_TTL:
                return response
            else:
                # Remove expired cache entry
                del REQUEST_CACHE[request_key]
        return None
    
    @staticmethod
    def cache_response(request_key: str, response: Any):
        """Cache response for deduplication"""
        REQUEST_CACHE[request_key] = (response, time.time())
        
        # Limit cache size (LRU-style cleanup)
        if len(REQUEST_CACHE) > 1000:
            oldest_keys = sorted(REQUEST_CACHE.keys(), 
                               key=lambda k: REQUEST_CACHE[k][1])[:100]
            for key in oldest_keys:
                del REQUEST_CACHE[key]


class SpeedOptimizedSearchEngine:
    """High-performance web search engine with speed optimizations"""
    
    def __init__(self):
        self.connection_pool = ConnectionPool(max_connections=50)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Pre-compile frequently used regex patterns
        self.compiled_patterns = SEARCH_TRIGGER_PATTERNS
        
        # Cache for search triggers to avoid repeated analysis
        self.trigger_cache: Dict[str, Tuple[bool, float, str]] = {}
        
        # Batch processing queue
        self.search_queue: List[Tuple[str, asyncio.Future]] = []
        self.batch_processor_task: Optional[asyncio.Task] = None
        
        print(f"{Fore.GREEN}⚡ Speed-Optimized Search Engine initialized{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Connection pooling: Up to 50 concurrent connections{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Request deduplication: 5-minute cache{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Compiled regex patterns: {len(self.compiled_patterns)} patterns{Style.RESET_ALL}")
    
    def analyze_search_need_fast(self, query: str) -> Tuple[bool, float, str]:
        """Optimized search need analysis using pre-compiled patterns"""
        # Check cache first
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        if query_hash in self.trigger_cache:
            return self.trigger_cache[query_hash]
        
        confidence = 0.0
        reasons = []
        
        # Use pre-compiled patterns for speed
        if self.compiled_patterns['high_confidence'].search(query):
            confidence = max(confidence, 0.9)
            reasons.append("high-priority pattern match")
        
        if self.compiled_patterns['medium_confidence'].search(query):
            confidence = max(confidence, 0.7)
            reasons.append("medium-priority pattern match")
        
        if self.compiled_patterns['academic'].search(query):
            confidence = max(confidence, 0.75)
            reasons.append("academic pattern match")
        
        if self.compiled_patterns['web_specific'].search(query):
            confidence = max(confidence, 0.8)
            reasons.append("web-specific pattern match")
        
        # Quick heuristics for entities (proper nouns)
        if any(word[0].isupper() and len(word) > 3 for word in query.split()):
            confidence = max(confidence, 0.6)
            reasons.append("entity detected")
        
        needs_search = confidence >= 0.7
        reason = "; ".join(reasons) if reasons else "no triggers"
        
        # Cache result
        result = (needs_search, confidence, reason)
        self.trigger_cache[query_hash] = result
        
        # Limit cache size
        if len(self.trigger_cache) > 500:
            # Remove oldest 100 entries (simple cleanup)
            keys_to_remove = list(self.trigger_cache.keys())[:100]
            for key in keys_to_remove:
                del self.trigger_cache[key]
        
        return result
    
    async def parallel_search_multiple_providers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search multiple providers in parallel using asyncio.gather"""
        session = await self.connection_pool.get_session()
        
        # Define search tasks for parallel execution
        search_tasks = [
            self._search_duckduckgo_fast(session, query),
            self._search_wikipedia_fast(session, query),
        ]
        
        # Execute all searches in parallel
        start_time = time.time()
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Collect successful results
        all_results = []
        successful_providers = []
        
        provider_names = ['DuckDuckGo', 'Wikipedia']
        for i, result in enumerate(results):
            if not isinstance(result, Exception) and result:
                all_results.extend(result)
                successful_providers.append(provider_names[i])
        
        print(f"{Fore.GREEN}⚡ Parallel search completed: {len(all_results)} results in {duration:.3f}s{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✅ Successful providers: {', '.join(successful_providers)}{Style.RESET_ALL}")
        
        # Return top results only
        return all_results[:max_results]
    
    async def _search_duckduckgo_fast(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Speed-optimized DuckDuckGo search"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            # Check for duplicate request
            request_key = RequestDeduplicator.get_request_key(url, params)
            cached = RequestDeduplicator.get_cached_response(request_key)
            if cached:
                return cached
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('Heading', query),
                            'url': data.get('AbstractURL', ''),
                            'snippet': data.get('Abstract', ''),
                            'source': 'DuckDuckGo',
                            'score': 0.9
                        })
                    
                    # Process related topics efficiently
                    for topic in data.get('RelatedTopics', [])[:3]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append({
                                'title': topic.get('Text', '')[:80],
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', ''),
                                'source': 'DuckDuckGo',
                                'score': 0.7
                            })
                    
                    # Cache result
                    RequestDeduplicator.cache_response(request_key, results)
                    return results
                    
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ DuckDuckGo search error: {str(e)[:50]}...{Style.RESET_ALL}")
        
        return []
    
    async def _search_wikipedia_fast(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Speed-optimized Wikipedia search"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': 3
            }
            
            request_key = RequestDeduplicator.get_request_key(url, params)
            cached = RequestDeduplicator.get_cached_response(request_key)
            if cached:
                return cached
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('query', {}).get('search', []):
                        results.append({
                            'title': item.get('title', ''),
                            'url': f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}",
                            'snippet': item.get('snippet', ''),
                            'source': 'Wikipedia',
                            'score': 0.85
                        })
                    
                    RequestDeduplicator.cache_response(request_key, results)
                    return results
                    
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Wikipedia search error: {str(e)[:50]}...{Style.RESET_ALL}")
        
        return []
    
    async def close(self):
        """Clean shutdown of optimized components"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
        
        self.thread_pool.shutdown(wait=False)
        await self.connection_pool.close()
        
        print(f"{Fore.CYAN}⚡ Speed-optimized search engine closed{Style.RESET_ALL}")


class SpeedOptimizedContentProcessor:
    """High-performance content processor with optimizations"""
    
    def __init__(self):
        # Pre-compile patterns for content cleaning
        self.cleaning_patterns = {
            'whitespace': re.compile(r'\s+'),
            'artifacts': re.compile(r'(Skip to main content|Accept cookies|Advertisement|\[Ad\])', re.IGNORECASE)
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
    async def process_results_parallel(self, search_results: List[Dict]) -> str:
        """Process search results in parallel"""
        if not search_results:
            return "No search results to process."
        
        start_time = time.time()
        
        # Process results in parallel using thread pool for CPU-bound tasks
        loop = asyncio.get_event_loop()
        
        processing_tasks = [
            loop.run_in_executor(self.thread_pool, self._process_single_result, result)
            for result in search_results[:5]  # Limit to 5 for speed
        ]
        
        processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in processed_results if not isinstance(r, Exception)]
        
        duration = time.time() - start_time
        print(f"{Fore.GREEN}⚡ Parallel content processing: {len(successful_results)} items in {duration:.3f}s{Style.RESET_ALL}")
        
        return self._create_ai_context_fast(successful_results, search_results[0].get('query', 'search'))
    
    def _process_single_result(self, result: Dict) -> Dict:
        """Process a single result (CPU-bound, runs in thread pool)"""
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        url = result.get('url', '')
        source = result.get('source', '')
        
        # Fast text cleaning using pre-compiled patterns
        clean_snippet = self.cleaning_patterns['artifacts'].sub('', snippet)
        clean_snippet = self.cleaning_patterns['whitespace'].sub(' ', clean_snippet).strip()
        
        # Quick key points extraction (simplified for speed)
        key_points = []
        if len(clean_snippet) > 100:
            sentences = clean_snippet.split('. ')
            # Take first 2 sentences as key points
            key_points = sentences[:2]
        else:
            key_points = [clean_snippet] if clean_snippet else []
        
        return {
            'title': title,
            'url': url,
            'snippet': clean_snippet,
            'source': source,
            'key_points': key_points,
            'word_count': len(clean_snippet.split())
        }
    
    def _create_ai_context_fast(self, processed_results: List[Dict], query: str) -> str:
        """Create AI context quickly using string operations"""
        if not processed_results:
            return f"No processed results for: {query}"
        
        context_parts = [
            f"Web search results for: {query}",
            f"Found {len(processed_results)} relevant sources:\n"
        ]
        
        for i, result in enumerate(processed_results, 1):
            context_parts.append(f"Source {i}: {result.get('source', 'Unknown')} - {result.get('title', 'No title')}")
            context_parts.append(f"URL: {result.get('url', 'No URL')}")
            context_parts.append(f"Content: {result.get('snippet', 'No content')}")
            
            if result.get('key_points'):
                context_parts.append("Key points: " + "; ".join(result['key_points']))
            
            context_parts.append("")  # Spacing
        
        context_parts.append("---")
        context_parts.append("Please provide a comprehensive response based on the above sources.")
        
        return "\n".join(context_parts)
    
    def cleanup(self):
        """Cleanup thread pool"""
        self.thread_pool.shutdown(wait=False)


class SpeedOptimizedStudyForgeAI:
    """Main speed-optimized StudyForge AI class"""
    
    def __init__(self):
        self.search_engine = SpeedOptimizedSearchEngine()
        self.content_processor = SpeedOptimizedContentProcessor()
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "gpt-oss:20b"
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'web_searches': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0
        }
        
        print(f"{Fore.LIGHTGREEN_EX}🚀 StudyForge AI - Speed Optimized Version Ready!{Style.RESET_ALL}")
    
    async def query_fast(self, user_query: str, force_web: bool = False) -> str:
        """Ultra-fast query processing with all optimizations"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        print(f"\n{Fore.LIGHTCYAN_EX}⚡ Processing: {user_query}{Style.RESET_ALL}")
        
        try:
            # Step 1: Fast search trigger analysis
            needs_web, confidence, reason = self.search_engine.analyze_search_need_fast(user_query)
            
            web_context = ""
            if force_web or needs_web:
                print(f"{Fore.YELLOW}🌐 Web search triggered: {reason} (confidence: {confidence:.2f}){Style.RESET_ALL}")
                
                # Step 2: Parallel web search
                search_results = await self.search_engine.parallel_search_multiple_providers(user_query)
                
                if search_results:
                    self.stats['web_searches'] += 1
                    
                    # Step 3: Parallel content processing
                    web_context = await self.content_processor.process_results_parallel(search_results)
                else:
                    web_context = "Web search completed but no results found."
            else:
                print(f"{Fore.BLUE}💭 Local knowledge only: {reason}{Style.RESET_ALL}")
            
            # Step 4: Query local AI model
            ai_response = await self._query_ollama_fast(user_query, web_context)
            
            # Update performance stats
            response_time = time.time() - start_time
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time) /
                self.stats['total_queries']
            )
            
            print(f"{Fore.GREEN}⚡ Response generated in {response_time:.3f}s{Style.RESET_ALL}")
            
            return ai_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            return error_msg
    
    async def _query_ollama_fast(self, query: str, context: str = "") -> str:
        """Speed-optimized Ollama query"""
        prompt = f"{context}\n\nUser: {query}\n\nAssistant:" if context else query
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096  # Optimized context window
            }
        }
        
        try:
            # Use optimized session from connection pool
            session = await self.search_engine.connection_pool.get_session()
            
            async with session.post(self.ollama_url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', 'No response from AI model.')
                else:
                    return f"AI model returned status {response.status}"
                    
        except asyncio.TimeoutError:
            return "AI processing timed out. Please try a simpler question."
        except Exception as e:
            return f"Error communicating with AI model: {str(e)}"
    
    def show_performance_stats(self):
        """Display performance statistics"""
        print(f"\n{Fore.CYAN}⚡ Performance Statistics:{Style.RESET_ALL}")
        print(f"   • Total queries: {self.stats['total_queries']}")
        print(f"   • Web searches: {self.stats['web_searches']}")
        print(f"   • Cache hits: {self.stats['cache_hits']}")
        print(f"   • Average response time: {self.stats['avg_response_time']:.3f}s")
        print(f"   • Cached requests: {len(REQUEST_CACHE)}")
        print(f"   • Trigger cache size: {len(self.search_engine.trigger_cache)}")
    
    async def run_speed_demo(self):
        """Run a speed demonstration"""
        print(f"\n{Fore.LIGHTGREEN_EX}🏃‍♂️ Speed Optimization Demo{Style.RESET_ALL}")
        print("="*50)
        
        demo_queries = [
            "What is Python programming?",
            "Latest AI developments 2024", 
            "How to implement machine learning?",
            "Current trends in software development"
        ]
        
        total_start = time.time()
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{Fore.LIGHTBLUE_EX}Query {i}/4: {query}{Style.RESET_ALL}")
            response = await self.query_fast(query)
            print(f"{Fore.GREEN}Response: {response[:100]}...{Style.RESET_ALL}")
        
        total_time = time.time() - total_start
        print(f"\n{Fore.LIGHTGREEN_EX}🎉 Demo completed in {total_time:.3f}s{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Average: {total_time/len(demo_queries):.3f}s per query{Style.RESET_ALL}")
        
        self.show_performance_stats()
    
    async def cleanup(self):
        """Cleanup all optimized components"""
        await self.search_engine.close()
        self.content_processor.cleanup()
        print(f"{Fore.GREEN}⚡ Speed-optimized StudyForge AI cleaned up{Style.RESET_ALL}")


async def main():
    """Speed-optimized main entry point"""
    print(f"{Fore.LIGHTGREEN_EX}🚀 StudyForge AI - Speed Optimized Version{Style.RESET_ALL}")
    print("="*60)
    
    ai = SpeedOptimizedStudyForgeAI()
    
    try:
        # Run speed demonstration
        await ai.run_speed_demo()
        
        print(f"\n{Fore.LIGHTCYAN_EX}💡 Speed Optimizations Implemented:{Style.RESET_ALL}")
        print("   ⚡ Pre-compiled regex patterns")
        print("   ⚡ Connection pooling & HTTP keep-alive")  
        print("   ⚡ Request deduplication cache")
        print("   ⚡ Parallel provider searches")
        print("   ⚡ Async content processing")
        print("   ⚡ Thread pool for CPU-bound tasks")
        print("   ⚡ Memory-efficient caching")
        print("   ⚡ Optimized timeout configurations")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Interrupted by user{Style.RESET_ALL}")
        return 1
    finally:
        await ai.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except Exception as e:
        print(f"{Fore.RED}💥 Fatal error: {e}{Style.RESET_ALL}")
        exit(1)