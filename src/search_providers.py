#!/usr/bin/env python3
"""
StudyForge AI - Search Providers
Specialized search implementations for different sources
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup, Comment
import xml.etree.ElementTree as ET

# Import from our web search engine
from web_search_engine import SearchResult

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


class BaseSearchProvider:
    """Base class for all search providers"""
    
    def __init__(self, config: Dict[str, Any] = None, session: aiohttp.ClientSession = None):
        self.config = config or {}
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = self.config.get('rate_limit', 1.0)
        
    async def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Main search method - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement search method")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.replace('www.', '')
        except:
            return "unknown"


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider (no API key required)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://api.duckduckgo.com/"
        self.html_search_url = "https://html.duckduckgo.com/html/"
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search using DuckDuckGo API and HTML interface"""
        await self._rate_limit()
        
        results = []
        
        # Try instant answer API first
        instant_results = await self._search_instant_answer(query)
        results.extend(instant_results)
        
        # Then try HTML search for more comprehensive results
        if len(results) < 5:  # Get more results if instant answer didn't provide enough
            html_results = await self._search_html(query, search_type)
            results.extend(html_results)
        
        return results[:self.config.get('max_results', 10)]
    
    async def _search_instant_answer(self, query: str) -> List[SearchResult]:
        """Use DuckDuckGo Instant Answer API"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                # Main abstract result
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
                
                # Answer (like calculations, definitions)
                if data.get('Answer'):
                    results.append(SearchResult(
                        title=f"Answer: {query}",
                        url=data.get('AnswerURL', ''),
                        snippet=data.get('Answer', ''),
                        source='DuckDuckGo',
                        timestamp=datetime.now(),
                        relevance_score=0.95,
                        content_type='answer'
                    ))
                
                # Related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(SearchResult(
                            title=topic.get('Text', '')[:80] + "...",
                            url=topic.get('FirstURL', ''),
                            snippet=topic.get('Text', ''),
                            source='DuckDuckGo',
                            timestamp=datetime.now(),
                            relevance_score=0.7,
                            content_type='related'
                        ))
                
                return results
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo instant answer failed: {e}")
            return []
    
    async def _search_html(self, query: str, search_type: str) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface (scraping)"""
        try:
            params = {
                'q': query,
                'b': '',  # Start from first result
                'kl': 'us-en',  # US English
                's': '0'  # Start position
            }
            
            headers = {
                'User-Agent': 'StudyForge-AI/1.0 Educational Bot',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            async with self.session.get(self.html_search_url, params=params, headers=headers) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                
                # Find result containers
                result_containers = soup.find_all('div', class_='web-result')
                
                for container in result_containers[:8]:  # Limit to first 8 results
                    try:
                        # Extract title and URL
                        title_elem = container.find('a', class_='result__a')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        
                        # Extract snippet
                        snippet_elem = container.find('a', class_='result__snippet')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if title and url:
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source='DuckDuckGo',
                                timestamp=datetime.now(),
                                relevance_score=0.8,
                                content_type='webpage'
                            ))
                    
                    except Exception as e:
                        self.logger.debug(f"Failed to parse result container: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []


class WikipediaProvider(BaseSearchProvider):
    """Wikipedia search provider"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.search_url = "https://en.wikipedia.org/w/api.php"
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search Wikipedia"""
        await self._rate_limit()
        
        try:
            # First, search for pages
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': self.config.get('max_results', 5)
            }
            
            async with self.session.get(self.search_url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get('query', {}).get('search', []):
                    try:
                        page_title = item['title']
                        page_url = f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}"
                        
                        # Get page summary
                        summary = await self._get_page_summary(page_title)
                        
                        results.append(SearchResult(
                            title=page_title,
                            url=page_url,
                            snippet=summary or item.get('snippet', ''),
                            source='Wikipedia',
                            timestamp=datetime.now(),
                            relevance_score=0.85,
                            content_type='encyclopedia'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to process Wikipedia result: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return []
    
    async def _get_page_summary(self, title: str) -> str:
        """Get Wikipedia page summary"""
        try:
            url = f"{self.api_url}{quote_plus(title)}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('extract', '')
        except Exception:
            pass
        return ""


class ArxivProvider(BaseSearchProvider):
    """ArXiv academic paper search provider"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "http://export.arxiv.org/api/query"
        self.min_request_interval = 3.0  # ArXiv recommends 3 seconds between requests
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search ArXiv for academic papers"""
        await self._rate_limit()
        
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': self.config.get('max_results', 10),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    return []
                
                xml_content = await response.text()
                root = ET.fromstring(xml_content)
                
                # Define namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                results = []
                entries = root.findall('.//atom:entry', ns)
                
                for entry in entries:
                    try:
                        title = entry.find('atom:title', ns).text.strip()
                        summary = entry.find('atom:summary', ns).text.strip()
                        
                        # Get paper URL
                        links = entry.findall('atom:link', ns)
                        paper_url = ""
                        for link in links:
                            if link.get('type') == 'text/html':
                                paper_url = link.get('href', '')
                                break
                        
                        # Get authors
                        authors = []
                        author_elems = entry.findall('atom:author', ns)
                        for author in author_elems[:3]:  # Limit to first 3 authors
                            name = author.find('atom:name', ns)
                            if name is not None:
                                authors.append(name.text)
                        
                        author_str = ", ".join(authors)
                        if len(authors) > 3:
                            author_str += " et al."
                        
                        # Get publication date
                        published = entry.find('atom:published', ns)
                        pub_date = published.text if published is not None else ""
                        
                        # Create enhanced title with authors and date
                        enhanced_title = f"{title}"
                        if author_str:
                            enhanced_title += f" - {author_str}"
                        if pub_date:
                            try:
                                date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                enhanced_title += f" ({date_obj.year})"
                            except:
                                pass
                        
                        results.append(SearchResult(
                            title=enhanced_title,
                            url=paper_url,
                            snippet=summary[:300] + "...",
                            source='ArXiv',
                            timestamp=datetime.now(),
                            relevance_score=0.9,
                            content_type='academic'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to parse ArXiv entry: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
            return []


class RedditProvider(BaseSearchProvider):
    """Reddit search provider for community discussions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://www.reddit.com/search.json"
        self.min_request_interval = 2.0  # Reddit rate limiting
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search Reddit"""
        await self._rate_limit()
        
        try:
            params = {
                'q': query,
                'sort': 'relevance',
                'limit': self.config.get('max_results', 8),
                't': 'month',  # Last month
                'type': 'link'
            }
            
            headers = {
                'User-Agent': 'StudyForge-AI/1.0 Educational Research Bot'
            }
            
            async with self.session.get(self.base_url, params=params, headers=headers) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get('data', {}).get('children', []):
                    try:
                        post = item['data']
                        
                        title = post.get('title', '')
                        url = f"https://reddit.com{post.get('permalink', '')}"
                        subreddit = post.get('subreddit', '')
                        score = post.get('score', 0)
                        num_comments = post.get('num_comments', 0)
                        
                        # Create snippet with metadata
                        snippet = f"r/{subreddit} • {score} upvotes • {num_comments} comments"
                        if post.get('selftext'):
                            snippet += f" • {post['selftext'][:200]}..."
                        
                        # Calculate relevance based on score and comments
                        relevance = min(0.9, 0.6 + (score / 1000) * 0.2 + (num_comments / 100) * 0.1)
                        
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source='Reddit',
                            timestamp=datetime.now(),
                            relevance_score=relevance,
                            content_type='discussion'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to parse Reddit result: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Reddit search failed: {e}")
            return []


class GitHubProvider(BaseSearchProvider):
    """GitHub code and repository search provider"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_url = "https://api.github.com/search/"
        self.min_request_interval = 1.5  # GitHub rate limiting
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search GitHub repositories and code"""
        await self._rate_limit()
        
        results = []
        
        # Search repositories
        if search_type in ['general', 'code', 'repositories']:
            repo_results = await self._search_repositories(query)
            results.extend(repo_results)
        
        # Search code if specifically requested
        if search_type == 'code' and len(results) < 5:
            code_results = await self._search_code(query)
            results.extend(code_results)
        
        return results[:self.config.get('max_results', 10)]
    
    async def _search_repositories(self, query: str) -> List[SearchResult]:
        """Search GitHub repositories"""
        try:
            url = f"{self.search_url}repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 8
            }
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'StudyForge-AI/1.0'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for repo in data.get('items', []):
                    try:
                        name = repo.get('full_name', '')
                        html_url = repo.get('html_url', '')
                        description = repo.get('description', 'No description provided')
                        stars = repo.get('stargazers_count', 0)
                        language = repo.get('language', 'Unknown')
                        
                        # Create snippet with metadata
                        snippet = f"{description} • ⭐ {stars} stars • Language: {language}"
                        
                        # Calculate relevance based on stars
                        relevance = min(0.9, 0.7 + (stars / 10000) * 0.2)
                        
                        results.append(SearchResult(
                            title=name,
                            url=html_url,
                            snippet=snippet,
                            source='GitHub',
                            timestamp=datetime.now(),
                            relevance_score=relevance,
                            content_type='repository'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to parse GitHub repository: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"GitHub repository search failed: {e}")
            return []
    
    async def _search_code(self, query: str) -> List[SearchResult]:
        """Search GitHub code"""
        try:
            url = f"{self.search_url}code"
            params = {
                'q': query,
                'sort': 'indexed',
                'order': 'desc',
                'per_page': 5
            }
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'StudyForge-AI/1.0'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for code_item in data.get('items', []):
                    try:
                        name = code_item.get('name', '')
                        path = code_item.get('path', '')
                        html_url = code_item.get('html_url', '')
                        repo_name = code_item.get('repository', {}).get('full_name', '')
                        
                        title = f"{name} - {repo_name}"
                        snippet = f"Code file: {path}"
                        
                        results.append(SearchResult(
                            title=title,
                            url=html_url,
                            snippet=snippet,
                            source='GitHub Code',
                            timestamp=datetime.now(),
                            relevance_score=0.8,
                            content_type='code'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to parse GitHub code result: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"GitHub code search failed: {e}")
            return []


class NewsAPIProvider(BaseSearchProvider):
    """News API provider for current news (requires API key)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://newsapi.org/v2/everything"
        self.min_request_interval = 1.0
    
    async def search(self, query: str, search_type: str = "general", **kwargs) -> List[SearchResult]:
        """Search news articles"""
        api_key = self.config.get('news_api_key')
        if not api_key:
            return []
        
        await self._rate_limit()
        
        try:
            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            params = {
                'q': query,
                'apiKey': api_key,
                'sortBy': 'relevancy',
                'pageSize': self.config.get('max_results', 10),
                'language': 'en',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for article in data.get('articles', []):
                    try:
                        title = article.get('title', '')
                        url = article.get('url', '')
                        description = article.get('description', '')
                        source_name = article.get('source', {}).get('name', 'News')
                        published_at = article.get('publishedAt', '')
                        
                        # Parse publication date
                        pub_date = ""
                        try:
                            date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            pub_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            pass
                        
                        # Create snippet with date
                        snippet = description
                        if pub_date:
                            snippet += f" (Published: {pub_date})"
                        
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source=source_name,
                            timestamp=datetime.now(),
                            relevance_score=0.85,
                            content_type='news'
                        ))
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to parse news article: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"News API search failed: {e}")
            return []


# Provider registry for easy management
AVAILABLE_PROVIDERS = {
    'duckduckgo': DuckDuckGoProvider,
    'wikipedia': WikipediaProvider,
    'arxiv': ArxivProvider,
    'reddit': RedditProvider,
    'github': GitHubProvider,
    'newsapi': NewsAPIProvider
}


def get_provider(provider_name: str, config: Dict[str, Any] = None, 
                session: aiohttp.ClientSession = None) -> BaseSearchProvider:
    """Get a search provider instance"""
    if provider_name not in AVAILABLE_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    provider_class = AVAILABLE_PROVIDERS[provider_name]
    return provider_class(config=config, session=session)


def get_providers_for_search_type(search_type: str) -> List[str]:
    """Get recommended providers for a specific search type"""
    provider_map = {
        'general': ['duckduckgo', 'wikipedia'],
        'academic': ['arxiv', 'wikipedia'],
        'news': ['newsapi', 'reddit'],
        'code': ['github', 'duckduckgo'],
        'discussion': ['reddit'],
        'research': ['arxiv', 'wikipedia']
    }
    
    return provider_map.get(search_type, ['duckduckgo'])


if __name__ == "__main__":
    async def test_providers():
        """Test all available providers"""
        
        # Create session
        async with aiohttp.ClientSession() as session:
            test_query = "machine learning"
            
            print(f"🧪 Testing search providers with query: '{test_query}'")
            print("="*60)
            
            for provider_name, provider_class in AVAILABLE_PROVIDERS.items():
                print(f"\n🔍 Testing {provider_name}...")
                
                try:
                    provider = provider_class(
                        config={'max_results': 3},
                        session=session
                    )
                    
                    results = await provider.search(test_query)
                    
                    if results:
                        print(f"✅ {provider_name}: {len(results)} results")
                        for i, result in enumerate(results[:2], 1):
                            print(f"  {i}. {result.title[:60]}...")
                            print(f"     {result.url}")
                    else:
                        print(f"⚠️  {provider_name}: No results")
                        
                except Exception as e:
                    print(f"❌ {provider_name}: {str(e)}")
                
                # Small delay between providers
                await asyncio.sleep(1)
    
    asyncio.run(test_providers())