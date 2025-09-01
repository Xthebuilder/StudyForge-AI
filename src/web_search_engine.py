#!/usr/bin/env python3
"""
StudyForge AI - Web Search Engine
Enterprise-grade web search with timeout management and multi-provider support
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus, urljoin
import hashlib
import logging
from pathlib import Path

# Import database manager
from .database_manager import DatabaseManager
from .memory_compressor import MemoryCompressor

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
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0
    content_type: str = "webpage"  # webpage, academic, news, code


@dataclass
class SearchResponse:
    """Complete search response with metadata"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time: float
    providers_used: List[str]
    cached: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SearchCache:
    """Intelligent caching system for search results"""
    
    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.logger = logging.getLogger(f"{__name__}.SearchCache")
    
    def _get_cache_key(self, query: str, search_type: str = "general") -> str:
        """Generate cache key for query"""
        combined = f"{query.lower().strip()}_{search_type}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"search_{cache_key}.json"
    
    def get(self, query: str, search_type: str = "general") -> Optional[SearchResponse]:
        """Get cached search results"""
        try:
            cache_key = self._get_cache_key(query, search_type)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.max_age:
                cache_path.unlink()  # Remove expired cache
                return None
            
            # Reconstruct SearchResponse
            results = [SearchResult(**result) for result in data['results']]
            response = SearchResponse(
                query=data['query'],
                results=results,
                total_found=data['total_found'],
                search_time=data['search_time'],
                providers_used=data['providers_used'],
                cached=True,
                timestamp=cached_time
            )
            
            self.logger.info(f"Cache hit for query: {query[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, response: SearchResponse):
        """Cache search results"""
        try:
            cache_key = self._get_cache_key(response.query)
            cache_path = self._get_cache_path(cache_key)
            
            # Convert to serializable format
            data = {
                'query': response.query,
                'results': [asdict(result) for result in response.results],
                'total_found': response.total_found,
                'search_time': response.search_time,
                'providers_used': response.providers_used,
                'timestamp': response.timestamp.isoformat()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cached search results for: {response.query[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Cache write error: {e}")
    
    def clear_expired(self):
        """Clear expired cache entries"""
        try:
            now = datetime.now()
            removed_count = 0
            
            for cache_file in self.cache_dir.glob("search_*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if now - cached_time > self.max_age:
                        cache_file.unlink()
                        removed_count += 1
                        
                except Exception:
                    # Remove corrupted cache files
                    cache_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} expired cache entries")
                
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")


class WebSearchEngine:
    """Enterprise web search engine with multi-provider support"""
    
    def __init__(self, config: Dict[str, Any] = None, db_manager: DatabaseManager = None):
        self.config = config or self._default_config()
        self.cache = SearchCache(
            cache_dir=self.config.get('cache_dir', 'search_cache'),
            max_age_hours=self.config.get('cache_max_age_hours', 24)
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.WebSearchEngine")
        
        # Database integration
        self.db_manager = db_manager or DatabaseManager()
        self.memory_compressor = MemoryCompressor()
        
        # Search provider configurations
        self.providers = {
            'duckduckgo': {
                'enabled': True,
                'priority': 1,
                'timeout': 10,
                'rate_limit': 1.0  # seconds between requests
            },
            'bing': {
                'enabled': self.config.get('bing_api_key') is not None,
                'priority': 2,
                'timeout': 15,
                'rate_limit': 0.5
            },
            'google': {
                'enabled': self.config.get('google_api_key') is not None,
                'priority': 3,
                'timeout': 15,
                'rate_limit': 0.5
            }
        }
        
        # Search trigger phrases for intelligent detection
        self.search_triggers = {
            'recent': ['latest', 'recent', 'current', 'new', 'today', 'this week', 'this month'],
            'temporal': ['2024', '2023', 'yesterday', 'last week', 'last month', 'breaking'],
            'web_specific': ['website', 'online', 'internet', 'web', 'url', 'link'],
            'news': ['news', 'headlines', 'breaking', 'update', 'announcement'],
            'trends': ['trending', 'popular', 'viral', 'hot', 'buzz'],
            'research': ['study', 'research', 'paper', 'publication', 'journal']
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default search engine configuration"""
        return {
            'max_results_per_provider': 10,
            'total_max_results': 30,
            'request_timeout': 30,
            'total_timeout': 120,
            'retry_attempts': 3,
            'retry_delay': 2.0,
            'user_agent': 'StudyForge-AI/1.0 (Educational Research Bot)',
            'enable_caching': True,
            'cache_max_age_hours': 24,
            'respect_robots_txt': True,
            'rate_limit_delay': 1.0,
            # API keys (set via environment variables)
            'google_api_key': None,
            'google_search_engine_id': None,
            'bing_api_key': None
        }
    
    async def create_session(self):
        """Create HTTP session with timeout configuration"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                total=self.config['total_timeout'],
                connect=self.config['request_timeout'] // 3,
                sock_read=self.config['request_timeout']
            )
            
            headers = {
                'User-Agent': self.config.get('user_agent', 'StudyForge-AI/1.0'),
                'Accept': 'application/json, text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        
        return self.session
    
    def should_search_web(self, query: str) -> Tuple[bool, str]:
        """Intelligent detection if query needs web search"""
        query_lower = query.lower()
        
        # Check for temporal triggers (most likely to need web search)
        for trigger in self.search_triggers['temporal']:
            if trigger in query_lower:
                return True, f"temporal indicator: '{trigger}'"
        
        # Check for recent/current information requests
        for trigger in self.search_triggers['recent']:
            if trigger in query_lower:
                return True, f"recent information request: '{trigger}'"
        
        # Check for news/updates
        for trigger in self.search_triggers['news']:
            if trigger in query_lower:
                return True, f"news request: '{trigger}'"
        
        # Check for trending topics
        for trigger in self.search_triggers['trends']:
            if trigger in query_lower:
                return True, f"trending topic: '{trigger}'"
        
        # Check for research/academic queries
        for trigger in self.search_triggers['research']:
            if trigger in query_lower:
                return True, f"research query: '{trigger}'"
        
        # Check for web-specific requests
        for trigger in self.search_triggers['web_specific']:
            if trigger in query_lower:
                return True, f"web-specific request: '{trigger}'"
        
        # Questions about specific companies, products, or people (likely need current info)
        web_indicators = [
            'what happened to', 'what is', 'who is', 'where is', 
            'how to', 'when did', 'why did', 'price of', 'cost of',
            'review of', 'opinion on'
        ]
        
        for indicator in web_indicators:
            if indicator in query_lower:
                return True, f"factual query: '{indicator}'"
        
        return False, "no web search triggers detected"
    
    async def search(self, query: str, search_type: str = "general", force_web: bool = False) -> SearchResponse:
        """Main search method with intelligent provider selection"""
        start_time = time.time()
        
        print(f"{Fore.CYAN}🔍 StudyForge AI - Web Search Engine{Style.RESET_ALL}")
        print(f"Query: {query}")
        
        # Check if web search is needed (unless forced)
        if not force_web:
            should_search, reason = self.should_search_web(query)
            if not should_search:
                print(f"{Fore.YELLOW}💡 Skipping web search - {reason}{Style.RESET_ALL}")
                return SearchResponse(
                    query=query,
                    results=[],
                    total_found=0,
                    search_time=0.0,
                    providers_used=[],
                    cached=False
                )
            else:
                print(f"{Fore.GREEN}🌐 Web search triggered - {reason}{Style.RESET_ALL}")
        
        # Check database cache first
        if self.config['enable_caching']:
            # Try database cache first (more persistent)
            cached_results = self.db_manager.get_cached_search(query, 'web_search')
            if cached_results:
                print(f"{Fore.MAGENTA}🗄️ Using database cached results ({len(cached_results)} items){Style.RESET_ALL}")
                return SearchResponse(
                    query=query,
                    results=[SearchResult(**result) for result in cached_results],
                    total_found=len(cached_results),
                    search_time=0.0,
                    providers_used=['cache'],
                    cached=True
                )
            
            # Fall back to file cache
            cached_response = self.cache.get(query, search_type)
            if cached_response:
                print(f"{Fore.MAGENTA}📦 Using file cached results ({len(cached_response.results)} items){Style.RESET_ALL}")
                return cached_response
        
        # Ensure session is created
        await self.create_session()
        
        # Get active providers sorted by priority
        active_providers = [
            (name, config) for name, config in self.providers.items()
            if config['enabled']
        ]
        active_providers.sort(key=lambda x: x[1]['priority'])
        
        print(f"{Fore.BLUE}🔄 Searching with providers: {[p[0] for p in active_providers]}{Style.RESET_ALL}")
        
        all_results = []
        used_providers = []
        
        # Search with each provider
        for provider_name, provider_config in active_providers:
            try:
                print(f"{Fore.CYAN}  → Searching {provider_name}...{Style.RESET_ALL}")
                
                if provider_name == 'duckduckgo':
                    results = await self._search_duckduckgo(query, search_type)
                elif provider_name == 'bing':
                    results = await self._search_bing(query, search_type)
                elif provider_name == 'google':
                    results = await self._search_google(query, search_type)
                else:
                    continue
                
                if results:
                    all_results.extend(results)
                    used_providers.append(provider_name)
                    print(f"{Fore.GREEN}  ✅ {provider_name}: {len(results)} results{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}  ⚠️  {provider_name}: No results{Style.RESET_ALL}")
                
                # Rate limiting
                await asyncio.sleep(provider_config['rate_limit'])
                
            except Exception as e:
                print(f"{Fore.RED}  ❌ {provider_name} failed: {str(e)[:50]}...{Style.RESET_ALL}")
                self.logger.error(f"Provider {provider_name} failed: {e}")
                continue
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_and_rank(all_results)
        
        # Limit results
        max_results = self.config['total_max_results']
        final_results = unique_results[:max_results]
        
        # Create response
        search_time = time.time() - start_time
        response = SearchResponse(
            query=query,
            results=final_results,
            total_found=len(unique_results),
            search_time=search_time,
            providers_used=used_providers,
            cached=False
        )
        
        # Cache results in both systems
        if self.config['enable_caching'] and final_results:
            # Cache in file system
            self.cache.set(response)
            
            # Cache in database with compression
            search_context = self.memory_compressor.compress_search_context(
                [asdict(result) for result in final_results]
            )
            self.db_manager.cache_search_result(query, 'web_search', [asdict(result) for result in final_results])
        
        print(f"{Fore.GREEN}✅ Search completed: {len(final_results)} results in {search_time:.2f}s{Style.RESET_ALL}")
        
        return response
    
    async def _search_duckduckgo(self, query: str, search_type: str) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Parse DuckDuckGo results
                    if data.get('Abstract'):
                        results.append(SearchResult(
                            title=data.get('Heading', query),
                            url=data.get('AbstractURL', ''),
                            snippet=data.get('Abstract', ''),
                            source='DuckDuckGo',
                            timestamp=datetime.now(),
                            relevance_score=0.9,
                            content_type='summary'
                        ))
                    
                    # Related topics
                    for topic in data.get('RelatedTopics', [])[:5]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append(SearchResult(
                                title=topic.get('Text', '')[:100],
                                url=topic.get('FirstURL', ''),
                                snippet=topic.get('Text', ''),
                                source='DuckDuckGo',
                                timestamp=datetime.now(),
                                relevance_score=0.7,
                                content_type='related'
                            ))
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            
        return []
    
    async def _search_bing(self, query: str, search_type: str) -> List[SearchResult]:
        """Search using Bing Search API (requires API key)"""
        api_key = self.config.get('bing_api_key')
        if not api_key:
            return []
        
        try:
            url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            params = {
                'q': query,
                'count': self.config['max_results_per_provider'],
                'offset': 0,
                'mkt': 'en-US',
                'safesearch': 'Moderate'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('webPages', {}).get('value', []):
                        results.append(SearchResult(
                            title=item.get('name', ''),
                            url=item.get('url', ''),
                            snippet=item.get('snippet', ''),
                            source='Bing',
                            timestamp=datetime.now(),
                            relevance_score=0.8,
                            content_type='webpage'
                        ))
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"Bing search failed: {e}")
            
        return []
    
    async def _search_google(self, query: str, search_type: str) -> List[SearchResult]:
        """Search using Google Custom Search API (requires API key)"""
        api_key = self.config.get('google_api_key')
        search_engine_id = self.config.get('google_search_engine_id')
        
        if not api_key or not search_engine_id:
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': self.config['max_results_per_provider']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('items', []):
                        results.append(SearchResult(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source='Google',
                            timestamp=datetime.now(),
                            relevance_score=0.9,
                            content_type='webpage'
                        ))
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            
        return []
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicates and rank results by relevance"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            # Simple URL deduplication
            if result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance score (descending)
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return unique_results
    
    async def close(self):
        """Close HTTP session and cleanup"""
        if self.session:
            await self.session.close()
            self.session = None
        
        # Clean up expired cache
        self.cache.clear_expired()
        
        print(f"{Fore.CYAN}🔒 Web Search Engine closed{Style.RESET_ALL}")


# Utility functions for easy integration
async def quick_search(query: str, force_web: bool = False) -> SearchResponse:
    """Quick search function for simple use cases"""
    engine = WebSearchEngine()
    try:
        return await engine.search(query, force_web=force_web)
    finally:
        await engine.close()


def format_search_results(response: SearchResponse, max_display: int = 5) -> str:
    """Format search results for display"""
    if not response.results:
        return f"No web results found for: {response.query}"
    
    output = []
    output.append(f"🌐 Web Search Results for: {response.query}")
    output.append(f"Found {response.total_found} results in {response.search_time:.2f}s")
    
    if response.cached:
        output.append(f"📦 (Using cached results)")
    
    output.append("")
    
    for i, result in enumerate(response.results[:max_display], 1):
        output.append(f"{i}. 📄 {result.title}")
        output.append(f"   🔗 {result.url}")
        output.append(f"   💬 {result.snippet}")
        output.append(f"   📡 Source: {result.source}")
        output.append("")
    
    if len(response.results) > max_display:
        output.append(f"... and {len(response.results) - max_display} more results")
    
    return "\n".join(output)


if __name__ == "__main__":
    async def test_search():
        engine = WebSearchEngine()
        try:
            # Test intelligent search detection
            test_queries = [
                "What is Python?",  # Should not trigger web search
                "Latest Python 3.12 features",  # Should trigger web search
                "Breaking news about AI today",  # Should trigger web search
                "How to implement binary search"  # May or may not trigger
            ]
            
            for query in test_queries:
                print(f"\n{'='*60}")
                print(f"Testing: {query}")
                print('='*60)
                
                should_search, reason = engine.should_search_web(query)
                print(f"Should search web: {should_search} ({reason})")
                
                if should_search:
                    response = await engine.search(query)
                    print(format_search_results(response, max_display=3))
        
        finally:
            await engine.close()
    
    asyncio.run(test_search())