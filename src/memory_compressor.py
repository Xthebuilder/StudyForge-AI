"""
Memory Compression System for StudyForge AI
Implements intelligent summarization and context compression
"""
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CompressionConfig:
    max_summary_length: int = 200
    key_topic_weight: float = 1.5
    recency_weight: float = 1.2
    importance_threshold: float = 0.6
    compression_ratio: float = 0.3  # Keep 30% of original content

class MemoryCompressor:
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for content analysis"""
        self.question_pattern = re.compile(r'\b(what|how|why|when|where|which|who|can|could|would|should)\b', re.IGNORECASE)
        self.topic_patterns = {
            'academic': re.compile(r'\b(study|research|assignment|project|exam|course|class|homework|paper|thesis|dissertation)\b', re.IGNORECASE),
            'technical': re.compile(r'\b(code|programming|software|algorithm|database|api|function|class|method|bug|error)\b', re.IGNORECASE),
            'search': re.compile(r'\b(search|find|look|information|data|results|source|reference)\b', re.IGNORECASE),
            'problem': re.compile(r'\b(problem|issue|error|help|fix|solve|trouble|difficulty)\b', re.IGNORECASE)
        }
        self.important_phrases = re.compile(r'\b(important|critical|urgent|deadline|due|remember|note|key|main|primary)\b', re.IGNORECASE)
    
    def compress_conversation_batch(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compress a batch of conversations into a summary"""
        if not conversations:
            return {'summary': '', 'key_topics': [], 'importance_score': 0.0}
        
        # Analyze conversations
        analysis = self._analyze_conversations(conversations)
        
        # Extract key information
        key_topics = self._extract_key_topics(conversations, analysis)
        important_exchanges = self._find_important_exchanges(conversations)
        
        # Generate compressed summary
        summary = self._generate_summary(conversations, key_topics, important_exchanges, analysis)
        
        return {
            'summary': summary,
            'key_topics': key_topics,
            'important_exchanges': important_exchanges[:3],  # Keep top 3
            'conversation_count': len(conversations),
            'time_span': self._get_time_span(conversations),
            'importance_score': analysis['avg_importance'],
            'compression_metadata': {
                'original_length': sum(len(c.get('content', '')) for c in conversations),
                'compressed_length': len(summary),
                'compression_ratio': len(summary) / max(sum(len(c.get('content', '')) for c in conversations), 1)
            }
        }
    
    def _analyze_conversations(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns and content"""
        analysis = {
            'total_messages': len(conversations),
            'user_messages': 0,
            'ai_messages': 0,
            'question_count': 0,
            'topic_distribution': {},
            'avg_importance': 0.0,
            'time_distribution': {}
        }
        
        total_importance = 0
        
        for conv in conversations:
            content = conv.get('content', '')
            msg_type = conv.get('message_type', 'unknown')
            importance = conv.get('importance_score', 0.5)
            
            total_importance += importance
            
            if msg_type == 'user':
                analysis['user_messages'] += 1
                # Count questions
                if self.question_pattern.search(content):
                    analysis['question_count'] += 1
            elif msg_type == 'ai':
                analysis['ai_messages'] += 1
            
            # Analyze topics
            for topic, pattern in self.topic_patterns.items():
                if pattern.search(content):
                    analysis['topic_distribution'][topic] = analysis['topic_distribution'].get(topic, 0) + 1
        
        analysis['avg_importance'] = total_importance / max(len(conversations), 1)
        
        return analysis
    
    def _extract_key_topics(self, conversations: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
        """Extract key topics from conversations"""
        topic_scores = {}
        
        # Score topics based on frequency and importance
        for conv in conversations:
            content = conv.get('content', '')
            importance = conv.get('importance_score', 0.5)
            
            # Extract noun phrases and keywords
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            for word in words:
                if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say']:
                    topic_scores[word] = topic_scores.get(word, 0) + importance
        
        # Sort topics by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top topics
        return [topic for topic, score in sorted_topics[:8] if score > 0.5]
    
    def _find_important_exchanges(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find the most important conversation exchanges"""
        important_exchanges = []
        
        i = 0
        while i < len(conversations):
            conv = conversations[i]
            
            # Look for high-importance messages or question-answer pairs
            if (conv.get('importance_score', 0) > self.config.importance_threshold or 
                self.important_phrases.search(conv.get('content', '')) or
                self.question_pattern.search(conv.get('content', ''))):
                
                exchange = {
                    'timestamp': conv.get('timestamp'),
                    'importance': conv.get('importance_score', 0),
                    'messages': [conv]
                }
                
                # Look for follow-up messages
                j = i + 1
                while j < len(conversations) and j < i + 3:  # Check next 2 messages
                    next_conv = conversations[j]
                    time_diff = abs(conv.get('timestamp', 0) - next_conv.get('timestamp', 0))
                    
                    if time_diff < 300:  # Within 5 minutes
                        exchange['messages'].append(next_conv)
                        exchange['importance'] = max(exchange['importance'], next_conv.get('importance_score', 0))
                    j += 1
                
                important_exchanges.append(exchange)
                i = j
            else:
                i += 1
        
        # Sort by importance and return top exchanges
        important_exchanges.sort(key=lambda x: x['importance'], reverse=True)
        return important_exchanges
    
    def _generate_summary(self, conversations: List[Dict[str, Any]], key_topics: List[str], 
                         important_exchanges: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate a compressed summary of conversations"""
        summary_parts = []
        
        # Time context
        time_span = self._get_time_span(conversations)
        if time_span:
            summary_parts.append(f"Session from {time_span['start'].strftime('%m/%d %H:%M')} to {time_span['end'].strftime('%m/%d %H:%M')}")
        
        # Topic summary
        if key_topics:
            topic_str = ', '.join(key_topics[:5])
            summary_parts.append(f"Main topics: {topic_str}")
        
        # Activity summary
        user_msgs = analysis['user_messages']
        ai_msgs = analysis['ai_messages']
        questions = analysis['question_count']
        
        activity_parts = []
        if user_msgs > 0:
            activity_parts.append(f"{user_msgs} user messages")
        if questions > 0:
            activity_parts.append(f"{questions} questions")
        if ai_msgs > 0:
            activity_parts.append(f"{ai_msgs} AI responses")
        
        if activity_parts:
            summary_parts.append(f"Activity: {', '.join(activity_parts)}")
        
        # Important highlights
        if important_exchanges:
            highlights = []
            for exchange in important_exchanges[:2]:  # Top 2 exchanges
                if exchange['messages']:
                    first_msg = exchange['messages'][0]
                    content_preview = first_msg.get('content', '')[:80]
                    if len(content_preview) >= 80:
                        content_preview += "..."
                    highlights.append(content_preview)
            
            if highlights:
                summary_parts.append(f"Key discussions: {' | '.join(highlights)}")
        
        # Join all parts
        summary = '. '.join(summary_parts)
        
        # Ensure summary doesn't exceed max length
        if len(summary) > self.config.max_summary_length:
            summary = summary[:self.config.max_summary_length - 3] + "..."
        
        return summary
    
    def _get_time_span(self, conversations: List[Dict[str, Any]]) -> Optional[Dict[str, datetime]]:
        """Get time span of conversations"""
        if not conversations:
            return None
        
        timestamps = []
        for conv in conversations:
            timestamp = conv.get('timestamp')
            if timestamp:
                if isinstance(timestamp, (int, float)):
                    timestamps.append(datetime.fromtimestamp(timestamp))
                elif isinstance(timestamp, datetime):
                    timestamps.append(timestamp)
        
        if not timestamps:
            return None
        
        return {
            'start': min(timestamps),
            'end': max(timestamps)
        }
    
    def compress_search_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compress search results context for efficient storage"""
        if not search_results:
            return {'summary': 'No search results', 'key_sources': [], 'total_results': 0}
        
        # Extract key information
        sources = set()
        key_points = []
        total_results = len(search_results)
        
        for result in search_results[:10]:  # Process top 10 results
            # Extract source
            if 'url' in result:
                domain = re.sub(r'https?://(www\.)?', '', result['url']).split('/')[0]
                sources.add(domain)
            elif 'source' in result:
                sources.add(result['source'])
            
            # Extract key points
            if 'snippet' in result:
                snippet = result['snippet'][:100]
                if len(snippet) >= 100:
                    snippet += "..."
                key_points.append(snippet)
            elif 'description' in result:
                desc = result['description'][:100]
                if len(desc) >= 100:
                    desc += "..."
                key_points.append(desc)
        
        # Create compressed summary
        summary_parts = []
        if sources:
            source_list = ', '.join(list(sources)[:5])
            summary_parts.append(f"Sources: {source_list}")
        
        if key_points:
            summary_parts.append(f"Key findings: {' | '.join(key_points[:3])}")
        
        summary = '. '.join(summary_parts)
        
        return {
            'summary': summary,
            'key_sources': list(sources)[:10],
            'total_results': total_results,
            'compression_ratio': len(summary) / max(sum(len(str(r)) for r in search_results), 1)
        }
    
    def should_compress(self, content_length: int, age_days: int, importance_score: float) -> bool:
        """Determine if content should be compressed"""
        # Compress if content is old and low importance
        if age_days > 7 and importance_score < 0.5:
            return True
        
        # Compress if content is very old regardless of importance
        if age_days > 30:
            return True
        
        # Compress if content is very long
        if content_length > 2000:
            return True
        
        return False