"""
Database Manager for StudyForge AI
Handles SQLite database operations with rolling memory management
"""
import sqlite3
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from contextlib import contextmanager

@dataclass
class ConversationEntry:
    id: Optional[int]
    user_id: str
    message_type: str  # 'user', 'ai', 'system'
    content: str
    timestamp: datetime
    importance_score: float
    search_context: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None

@dataclass
class MemoryConfig:
    max_conversations: int = 1000
    max_age_days: int = 30
    importance_threshold: float = 0.3
    summary_trigger_length: int = 500
    rolling_window_size: int = 50

class DatabaseManager:
    def __init__(self, db_path: str = "studyforge.db", config: MemoryConfig = None):
        self.db_path = db_path
        self.config = config or MemoryConfig()
        self._lock = threading.Lock()
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Conversations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    search_context TEXT,
                    summary TEXT,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Search cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    query TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    results TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # User profiles table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    memory_settings TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    last_active REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Memory summaries table for compressed historical data
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    time_period_start REAL NOT NULL,
                    time_period_end REAL NOT NULL,
                    summary TEXT NOT NULL,
                    conversation_count INTEGER NOT NULL,
                    importance_score REAL NOT NULL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user_time ON conversations(user_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_conversations_importance ON conversations(importance_score)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_search_cache_hash ON search_cache(query_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_search_cache_timestamp ON search_cache(timestamp)')
            
            conn.commit()
    
    def add_conversation(self, user_id: str, message_type: str, content: str, 
                        search_context: Optional[Dict[str, Any]] = None) -> int:
        """Add a conversation entry with automatic importance scoring"""
        importance_score = self._calculate_importance(content, message_type, search_context)
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO conversations (user_id, message_type, content, timestamp, 
                                         importance_score, search_context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, message_type, content, time.time(),
                importance_score, json.dumps(search_context) if search_context else None
            ))
            conn.commit()
            entry_id = cursor.lastrowid
        
        # Trigger rolling memory management
        self._manage_rolling_memory(user_id)
        return entry_id
    
    def get_conversation_history(self, user_id: str, limit: int = None) -> List[ConversationEntry]:
        """Get conversation history with rolling memory applied"""
        query = '''
            SELECT * FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        '''
        params = [user_id]
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        entries = []
        for row in rows:
            entries.append(ConversationEntry(
                id=row['id'],
                user_id=row['user_id'],
                message_type=row['message_type'],
                content=row['content'],
                timestamp=datetime.fromtimestamp(row['timestamp']),
                importance_score=row['importance_score'],
                search_context=json.loads(row['search_context']) if row['search_context'] else None,
                summary=row['summary']
            ))
        
        return entries
    
    def get_contextual_memory(self, user_id: str, query: str = None) -> Dict[str, Any]:
        """Get relevant memory context for current conversation"""
        # Get recent high-importance conversations
        recent_limit = self.config.rolling_window_size
        recent_conversations = self.get_conversation_history(user_id, recent_limit)
        
        # Get relevant summaries
        summaries = self._get_relevant_summaries(user_id, query)
        
        # Build context
        context = {
            'recent_conversations': recent_conversations[:10],  # Last 10 recent
            'important_conversations': [c for c in recent_conversations if c.importance_score > 0.7][:5],
            'historical_summaries': summaries,
            'total_conversations': len(recent_conversations)
        }
        
        return context
    
    def cache_search_result(self, query: str, provider: str, results: List[Dict[str, Any]]):
        """Cache search results for reuse"""
        query_hash = hashlib.sha256(f"{query}:{provider}".encode()).hexdigest()
        
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO search_cache 
                (query_hash, query, provider, results, timestamp, access_count)
                VALUES (?, ?, ?, ?, ?, COALESCE(
                    (SELECT access_count + 1 FROM search_cache WHERE query_hash = ?), 1
                ))
            ''', (query_hash, query, provider, json.dumps(results), time.time(), query_hash))
            conn.commit()
    
    def get_cached_search(self, query: str, provider: str, max_age_hours: int = 24) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results if available and fresh"""
        query_hash = hashlib.sha256(f"{query}:{provider}".encode()).hexdigest()
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.get_connection() as conn:
            row = conn.execute('''
                SELECT results FROM search_cache 
                WHERE query_hash = ? AND timestamp > ?
            ''', (query_hash, cutoff_time)).fetchone()
            
            if row:
                # Update access count
                conn.execute('''
                    UPDATE search_cache SET access_count = access_count + 1
                    WHERE query_hash = ?
                ''', (query_hash,))
                conn.commit()
                
                return json.loads(row['results'])
        
        return None
    
    def _calculate_importance(self, content: str, message_type: str, search_context: Dict[str, Any] = None) -> float:
        """Calculate importance score for memory management"""
        base_score = 0.5
        
        # Message type importance
        type_weights = {
            'user': 0.7,
            'ai': 0.5,
            'system': 0.3
        }
        base_score = type_weights.get(message_type, 0.5)
        
        # Content length boost
        if len(content) > 200:
            base_score += 0.1
        
        # Search context boost
        if search_context:
            base_score += 0.2
        
        # Keyword importance
        important_keywords = [
            'error', 'problem', 'help', 'important', 'urgent', 'deadline',
            'project', 'assignment', 'exam', 'study', 'research'
        ]
        
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in important_keywords if keyword in content_lower)
        base_score += min(keyword_matches * 0.1, 0.3)
        
        # Normalize to 0-1 range
        return min(max(base_score, 0.0), 1.0)
    
    def _manage_rolling_memory(self, user_id: str):
        """Implement rolling memory management"""
        with self.get_connection() as conn:
            # Count total conversations
            count_row = conn.execute(
                'SELECT COUNT(*) as count FROM conversations WHERE user_id = ?', 
                (user_id,)
            ).fetchone()
            
            total_conversations = count_row['count']
            
            if total_conversations > self.config.max_conversations:
                # Get conversations to archive/summarize
                old_conversations = conn.execute('''
                    SELECT * FROM conversations 
                    WHERE user_id = ? 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                ''', (user_id, total_conversations - self.config.rolling_window_size)).fetchall()
                
                if old_conversations:
                    # Create summary of old conversations
                    self._create_memory_summary(user_id, old_conversations, conn)
                    
                    # Remove old conversations, keeping only the most important ones
                    # Sort by importance and keep only a small number of high-importance conversations
                    sorted_old = sorted(old_conversations, key=lambda x: x['importance_score'], reverse=True)
                    keep_important = 5  # Keep top 5 most important old conversations
                    
                    conversations_to_delete = sorted_old[keep_important:]  # Delete all but top 5
                    
                    if conversations_to_delete:
                        old_ids = [row['id'] for row in conversations_to_delete]
                        placeholders = ','.join('?' * len(old_ids))
                        conn.execute(f'DELETE FROM conversations WHERE id IN ({placeholders})', old_ids)
                        conn.commit()
            
            # Clean old search cache
            self._clean_search_cache(conn)
    
    def _create_memory_summary(self, user_id: str, conversations: List[sqlite3.Row], conn: sqlite3.Connection):
        """Create a summary of conversations being archived"""
        if not conversations:
            return
        
        # Group conversations by time periods (e.g., daily)
        time_groups = {}
        for conv in conversations:
            day_key = datetime.fromtimestamp(conv['timestamp']).date().isoformat()
            if day_key not in time_groups:
                time_groups[day_key] = []
            time_groups[day_key].append(conv)
        
        # Create summaries for each time period
        for day_key, day_conversations in time_groups.items():
            if len(day_conversations) < 3:  # Skip if too few conversations
                continue
            
            # Create summary text
            summary_parts = []
            user_messages = [c for c in day_conversations if c['message_type'] == 'user']
            ai_messages = [c for c in day_conversations if c['message_type'] == 'ai']
            
            if user_messages:
                summary_parts.append(f"User topics: {'; '.join([c['content'][:100] for c in user_messages[:3]])}")
            
            if ai_messages:
                summary_parts.append(f"AI responses covered: {'; '.join([c['content'][:100] for c in ai_messages[:3]])}")
            
            summary_text = " | ".join(summary_parts)
            
            # Calculate average importance
            avg_importance = sum(c['importance_score'] for c in day_conversations) / len(day_conversations)
            
            # Insert summary
            start_time = min(c['timestamp'] for c in day_conversations)
            end_time = max(c['timestamp'] for c in day_conversations)
            
            conn.execute('''
                INSERT INTO memory_summaries 
                (user_id, time_period_start, time_period_end, summary, 
                 conversation_count, importance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, start_time, end_time, summary_text, 
                  len(day_conversations), avg_importance))
    
    def _get_relevant_summaries(self, user_id: str, query: str = None) -> List[Dict[str, Any]]:
        """Get relevant historical summaries"""
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM memory_summaries 
                WHERE user_id = ? 
                ORDER BY importance_score DESC, time_period_end DESC
                LIMIT 5
            ''', (user_id,)).fetchall()
            
            summaries = []
            for row in rows:
                summaries.append({
                    'time_period': f"{datetime.fromtimestamp(row['time_period_start']).date()} to {datetime.fromtimestamp(row['time_period_end']).date()}",
                    'summary': row['summary'],
                    'conversation_count': row['conversation_count'],
                    'importance_score': row['importance_score']
                })
            
            return summaries
    
    def _clean_search_cache(self, conn: sqlite3.Connection):
        """Clean old search cache entries"""
        # Remove entries older than configured age
        cutoff_time = time.time() - (self.config.max_age_days * 24 * 3600)
        conn.execute('DELETE FROM search_cache WHERE timestamp < ?', (cutoff_time,))
        
        # Keep only most accessed entries if cache is too large
        cache_count = conn.execute('SELECT COUNT(*) as count FROM search_cache').fetchone()['count']
        max_cache_size = 10000  # Maximum cache entries
        
        if cache_count > max_cache_size:
            conn.execute('''
                DELETE FROM search_cache 
                WHERE id NOT IN (
                    SELECT id FROM search_cache 
                    ORDER BY access_count DESC, timestamp DESC 
                    LIMIT ?
                )
            ''', (max_cache_size // 2,))
        
        conn.commit()
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics and memory usage"""
        with self.get_connection() as conn:
            # Conversation stats
            conv_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_conversations,
                    AVG(importance_score) as avg_importance,
                    MAX(timestamp) as last_activity
                FROM conversations 
                WHERE user_id = ?
            ''', (user_id,)).fetchone()
            
            # Summary stats
            summary_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_summaries,
                    SUM(conversation_count) as summarized_conversations
                FROM memory_summaries 
                WHERE user_id = ?
            ''', (user_id,)).fetchone()
            
            return {
                'total_conversations': conv_stats['total_conversations'] or 0,
                'avg_importance': conv_stats['avg_importance'] or 0,
                'last_activity': datetime.fromtimestamp(conv_stats['last_activity']) if conv_stats['last_activity'] else None,
                'total_summaries': summary_stats['total_summaries'] or 0,
                'summarized_conversations': summary_stats['summarized_conversations'] or 0
            }