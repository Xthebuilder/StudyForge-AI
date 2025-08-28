#!/usr/bin/env python3
"""
StudyForge AI - Content Processor
Processes and summarizes web content for AI analysis
"""

import asyncio
import aiohttp
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse
import hashlib
import json

# Import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Import from our modules
from .web_search_engine import SearchResult, SearchResponse

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
class ProcessedContent:
    """Processed and cleaned content from web sources"""
    original_url: str
    title: str
    clean_text: str
    summary: str
    key_points: List[str]
    source: str
    content_type: str
    word_count: int
    reading_time_minutes: int
    extraction_success: bool = True
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ContentProcessor:
    """Processes web content for AI consumption"""
    
    def __init__(self, config: Dict[str, Any] = None, session: aiohttp.ClientSession = None):
        self.config = config or self._default_config()
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.ContentProcessor")
        
        # Content extraction patterns
        self.content_selectors = [
            'article', 'main', '.content', '.post-content', 
            '.entry-content', '.article-content', '.post-body',
            '[role="main"]', '.main-content', '#content'
        ]
        
        # Elements to remove
        self.remove_selectors = [
            'nav', 'header', 'footer', 'aside', '.sidebar',
            '.navigation', '.menu', '.ads', '.advertisement',
            '.social-share', '.comments', '.comment-section',
            'script', 'style', 'noscript', '.cookie-notice'
        ]
    
    def _default_config(self) -> Dict[str, Any]:
        """Default content processor configuration"""
        return {
            'max_content_length': 8000,  # Max characters to process
            'min_content_length': 100,   # Minimum meaningful content
            'summary_sentences': 3,      # Number of sentences in summary
            'max_key_points': 5,         # Maximum key points to extract
            'request_timeout': 15,       # Timeout for fetching content
            'max_concurrent_extractions': 5,  # Max parallel extractions
            'reading_speed_wpm': 200,    # Words per minute for reading time
            'enable_full_extraction': True,  # Extract full content vs snippets only
        }
    
    async def process_search_results(self, search_response: SearchResponse) -> List[ProcessedContent]:
        """Process all search results to extract content"""
        if not search_response.results:
            return []
        
        print(f"{Fore.CYAN}🔄 Processing {len(search_response.results)} search results...{Style.RESET_ALL}")
        
        # Create semaphore to limit concurrent extractions
        semaphore = asyncio.Semaphore(self.config['max_concurrent_extractions'])
        
        # Process results concurrently
        tasks = []
        for result in search_response.results:
            task = self._process_single_result(result, semaphore)
            tasks.append(task)
        
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful extractions
        successful_extractions = []
        for result in processed_results:
            if isinstance(result, ProcessedContent) and result.extraction_success:
                successful_extractions.append(result)
            elif isinstance(result, Exception):
                self.logger.debug(f"Content extraction failed: {result}")
        
        print(f"{Fore.GREEN}✅ Successfully processed {len(successful_extractions)} content items{Style.RESET_ALL}")
        
        return successful_extractions
    
    async def _process_single_result(self, result: SearchResult, semaphore: asyncio.Semaphore) -> ProcessedContent:
        """Process a single search result"""
        async with semaphore:
            try:
                # For some content types, we can skip full extraction
                if result.content_type in ['answer', 'summary', 'related']:
                    return self._create_processed_from_snippet(result)
                
                # For academic papers, news, and webpages, try full extraction
                if self.config['enable_full_extraction'] and result.url:
                    extracted_content = await self._extract_full_content(result.url)
                    if extracted_content:
                        return self._create_processed_content(
                            result, extracted_content['text'], extracted_content['title']
                        )
                
                # Fallback to snippet-based processing
                return self._create_processed_from_snippet(result)
                
            except Exception as e:
                self.logger.debug(f"Failed to process {result.url}: {e}")
                return ProcessedContent(
                    original_url=result.url,
                    title=result.title,
                    clean_text=result.snippet,
                    summary=result.snippet,
                    key_points=[result.snippet] if result.snippet else [],
                    source=result.source,
                    content_type=result.content_type,
                    word_count=len(result.snippet.split()) if result.snippet else 0,
                    reading_time_minutes=1,
                    extraction_success=False
                )
    
    def _create_processed_from_snippet(self, result: SearchResult) -> ProcessedContent:
        """Create processed content from search result snippet"""
        snippet = result.snippet or ""
        words = snippet.split()
        word_count = len(words)
        
        # Extract key points from snippet
        key_points = self._extract_key_points_from_text(snippet)
        
        return ProcessedContent(
            original_url=result.url,
            title=result.title,
            clean_text=snippet,
            summary=snippet,
            key_points=key_points,
            source=result.source,
            content_type=result.content_type,
            word_count=word_count,
            reading_time_minutes=max(1, word_count // self.config['reading_speed_wpm']),
            extraction_success=True
        )
    
    async def _extract_full_content(self, url: str) -> Optional[Dict[str, str]]:
        """Extract full content from a webpage"""
        if not self.session or not BS4_AVAILABLE:
            return None
        
        try:
            headers = {
                'User-Agent': 'StudyForge-AI/1.0 Educational Content Processor',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            
            async with self.session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200 or response.content_type != 'text/html':
                    return None
                
                html = await response.text()
                return self._parse_html_content(html, url)
                
        except Exception as e:
            self.logger.debug(f"Failed to extract content from {url}: {e}")
            return None
    
    def _parse_html_content(self, html: str, url: str) -> Dict[str, str]:
        """Parse HTML content to extract clean text"""
        if not BS4_AVAILABLE:
            return {"title": "", "text": ""}
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for selector in self.remove_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract title
        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        # Try to find main content using selectors
        main_content = None
        for selector in self.content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return {"title": title, "text": ""}
        
        # Extract and clean text
        text_content = self._extract_clean_text(main_content)
        
        return {"title": title, "text": text_content}
    
    def _extract_clean_text(self, element) -> str:
        """Extract clean text from BeautifulSoup element"""
        if not BS4_AVAILABLE:
            return ""
        
        # Get all text, preserving some structure
        text_parts = []
        
        for elem in element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
            text = elem.get_text(strip=True)
            if text and len(text) > 20:  # Only meaningful text
                text_parts.append(text)
        
        # If structured extraction didn't work, get all text
        if not text_parts:
            full_text = element.get_text()
            # Clean up whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            return full_text[:self.config['max_content_length']]
        
        # Join parts with double newlines for readability
        combined_text = '\n\n'.join(text_parts)
        
        # Limit length
        if len(combined_text) > self.config['max_content_length']:
            combined_text = combined_text[:self.config['max_content_length']] + "..."
        
        return combined_text
    
    def _create_processed_content(self, result: SearchResult, content: str, title: str = None) -> ProcessedContent:
        """Create ProcessedContent from extracted content"""
        # Use extracted title if available, otherwise use result title
        final_title = title or result.title
        
        # Clean content
        clean_text = self._clean_text(content)
        
        # Ensure minimum content length
        if len(clean_text) < self.config['min_content_length']:
            # Fallback to snippet if content is too short
            clean_text = result.snippet or clean_text
        
        # Generate summary
        summary = self._generate_summary(clean_text)
        
        # Extract key points
        key_points = self._extract_key_points_from_text(clean_text)
        
        # Calculate stats
        words = clean_text.split()
        word_count = len(words)
        reading_time = max(1, word_count // self.config['reading_speed_wpm'])
        
        return ProcessedContent(
            original_url=result.url,
            title=final_title,
            clean_text=clean_text,
            summary=summary,
            key_points=key_points,
            source=result.source,
            content_type=result.content_type,
            word_count=word_count,
            reading_time_minutes=reading_time,
            extraction_success=True
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common web artifacts
        artifacts = [
            r'Skip to main content',
            r'Accept cookies',
            r'This website uses cookies',
            r'Subscribe to our newsletter',
            r'Sign up for our newsletter',
            r'Follow us on',
            r'Share this article',
            r'Advertisement',
            r'\[Ad\]',
        ]
        
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)
        
        # Clean up remaining whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary from text"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.config['summary_sentences']:
            return text
        
        # Simple extractive summary - take first few sentences
        # In a more sophisticated version, we'd use scoring algorithms
        summary_sentences = sentences[:self.config['summary_sentences']]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be improved with NLTK
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _extract_key_points_from_text(self, text: str) -> List[str]:
        """Extract key points from text"""
        if not text:
            return []
        
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 2:
            return [text] if text else []
        
        # Score sentences based on various factors
        scored_sentences = []
        
        # Keywords that often indicate important information
        important_keywords = [
            'important', 'key', 'main', 'primary', 'essential', 'critical',
            'significant', 'notable', 'major', 'fundamental', 'crucial',
            'breakthrough', 'discovery', 'finding', 'result', 'conclusion'
        ]
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on length (not too short, not too long)
            if 50 <= len(sentence) <= 200:
                score += 1
            
            # Score based on important keywords
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Score based on position (first and last sentences often important)
            if sentence in sentences[:2] or sentence in sentences[-2:]:
                score += 1
            
            # Score based on numbers/statistics
            if re.search(r'\d+[%$]?|\d+\.\d+', sentence):
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        key_points = [sent[0] for sent in scored_sentences[:self.config['max_key_points']]]
        
        return key_points
    
    def create_ai_context(self, processed_content: List[ProcessedContent], 
                         query: str) -> str:
        """Create context for AI model from processed content"""
        if not processed_content:
            return f"No relevant web content found for: {query}"
        
        context_parts = []
        context_parts.append(f"Web search results for: {query}")
        context_parts.append(f"Found {len(processed_content)} relevant sources:")
        context_parts.append("")
        
        for i, content in enumerate(processed_content, 1):
            context_parts.append(f"Source {i}: {content.source} - {content.title}")
            context_parts.append(f"URL: {content.original_url}")
            context_parts.append(f"Summary: {content.summary}")
            
            if content.key_points:
                context_parts.append("Key Points:")
                for point in content.key_points:
                    context_parts.append(f"  • {point}")
            
            context_parts.append("")
        
        # Add instruction for AI
        context_parts.append("---")
        context_parts.append("Please analyze the above web search results and provide a comprehensive response that:")
        context_parts.append("1. Synthesizes information from multiple sources")
        context_parts.append("2. Includes relevant citations and links")
        context_parts.append("3. Provides current, accurate information")
        context_parts.append("4. Addresses the user's specific question")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_processing_stats(self, processed_content: List[ProcessedContent]) -> Dict[str, Any]:
        """Get statistics about processed content"""
        if not processed_content:
            return {"total_sources": 0}
        
        stats = {
            "total_sources": len(processed_content),
            "successful_extractions": sum(1 for c in processed_content if c.extraction_success),
            "total_words": sum(c.word_count for c in processed_content),
            "avg_reading_time": sum(c.reading_time_minutes for c in processed_content) / len(processed_content),
            "content_types": {},
            "sources": {}
        }
        
        # Count content types and sources
        for content in processed_content:
            stats["content_types"][content.content_type] = stats["content_types"].get(content.content_type, 0) + 1
            stats["sources"][content.source] = stats["sources"].get(content.source, 0) + 1
        
        return stats


# Utility functions for easy use
async def process_web_search(search_response: SearchResponse, 
                           session: aiohttp.ClientSession = None) -> Tuple[List[ProcessedContent], str]:
    """Process web search results and create AI context"""
    processor = ContentProcessor(session=session)
    processed_content = await processor.process_search_results(search_response)
    ai_context = processor.create_ai_context(processed_content, search_response.query)
    return processed_content, ai_context


def format_processed_results(processed_content: List[ProcessedContent]) -> str:
    """Format processed content for display"""
    if not processed_content:
        return "No content processed successfully."
    
    output = []
    output.append(f"📚 Processed {len(processed_content)} sources:")
    output.append("")
    
    total_words = sum(c.word_count for c in processed_content)
    total_reading_time = sum(c.reading_time_minutes for c in processed_content)
    
    output.append(f"📊 Total: {total_words} words, ~{total_reading_time} min reading time")
    output.append("")
    
    for i, content in enumerate(processed_content, 1):
        status = "✅" if content.extraction_success else "⚠️"
        output.append(f"{status} {i}. {content.title}")
        output.append(f"    📡 {content.source} • {content.word_count} words • {content.reading_time_minutes} min")
        output.append(f"    📝 {content.summary[:100]}...")
        output.append("")
    
    return "\n".join(output)


if __name__ == "__main__":
    async def test_content_processor():
        """Test content processing"""
        from web_search_engine import SearchResult, SearchResponse
        
        # Create test search results
        test_results = [
            SearchResult(
                title="Python Programming Guide",
                url="https://example.com/python-guide",
                snippet="Python is a high-level programming language known for its simplicity and readability.",
                source="TestSource",
                timestamp=datetime.now(),
                relevance_score=0.9,
                content_type="webpage"
            )
        ]
        
        test_response = SearchResponse(
            query="Python programming",
            results=test_results,
            total_found=1,
            search_time=1.5,
            providers_used=["test"]
        )
        
        # Test processing
        processor = ContentProcessor()
        processed = await processor.process_search_results(test_response)
        
        print("🧪 Content Processor Test Results:")
        print("="*50)
        print(format_processed_results(processed))
        
        if processed:
            ai_context = processor.create_ai_context(processed, test_response.query)
            print("\n📝 Generated AI Context:")
            print("-"*30)
            print(ai_context[:500] + "..." if len(ai_context) > 500 else ai_context)
    
    asyncio.run(test_content_processor())