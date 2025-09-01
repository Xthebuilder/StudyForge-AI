"""
Knowledge Graph Builder for StudyForge AI Enhanced Edition
Creates and manages knowledge graphs of learning concepts and relationships
"""

import sqlite3
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

# Try to import network analysis libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ networkx not installed. Install with: pip install networkx")

@dataclass
class ConceptNode:
    concept_id: str
    name: str
    category: str
    importance_score: float
    learning_level: str  # 'beginner', 'intermediate', 'advanced'
    related_conversations: List[int]
    prerequisites: List[str]
    dependents: List[str]
    last_accessed: datetime

@dataclass
class ConceptRelationship:
    source_concept: str
    target_concept: str
    relationship_type: str  # 'prerequisite', 'similar', 'part_of', 'used_in'
    strength: float
    evidence_count: int
    last_reinforced: datetime

class KnowledgeGraphBuilder:
    """Builds and maintains knowledge graphs of learning concepts"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.db_path = "knowledge_graph.db"
        self.logger = logging.getLogger(f"{__name__}.KnowledgeGraphBuilder")
        self._init_graph_database()
        
        # In-memory graph for fast operations
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self._load_graph_from_db()
    
    def _init_graph_database(self):
        """Initialize knowledge graph database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Concepts table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                importance_score REAL DEFAULT 0.5,
                learning_level TEXT DEFAULT 'intermediate',
                related_conversations TEXT,
                prerequisites TEXT,
                dependents TEXT,
                last_accessed REAL DEFAULT (julianday('now')),
                created_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Relationships table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_concept TEXT NOT NULL,
                target_concept TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL NOT NULL,
                evidence_count INTEGER DEFAULT 1,
                last_reinforced REAL DEFAULT (julianday('now')),
                created_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Concept occurrences (for tracking concept usage)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS concept_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                conversation_id INTEGER,
                context TEXT,
                timestamp REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_concept)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_concept)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_occurrences_concept ON concept_occurrences(concept_id)')
        
        conn.commit()
        conn.close()
    
    def _load_graph_from_db(self):
        """Load knowledge graph from database into memory"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Load concepts as nodes
            concepts = conn.execute('SELECT * FROM concepts').fetchall()
            for concept in concepts:
                self.graph.add_node(concept['concept_id'], 
                                  name=concept['name'],
                                  category=concept['category'],
                                  importance=concept['importance_score'],
                                  level=concept['learning_level'])
            
            # Load relationships as edges
            relationships = conn.execute('SELECT * FROM relationships').fetchall()
            for rel in relationships:
                self.graph.add_edge(rel['source_concept'], rel['target_concept'],
                                  relationship=rel['relationship_type'],
                                  strength=rel['strength'],
                                  evidence=rel['evidence_count'])
            
            conn.close()
            self.logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} concepts, {self.graph.number_of_edges()} relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge graph: {e}")
    
    async def update_from_conversation(self, user_query: str, ai_response: str, user_id: str) -> bool:
        """Update knowledge graph based on conversation"""
        try:
            # Extract concepts from both query and response
            query_concepts = self._extract_concepts(user_query)
            response_concepts = self._extract_concepts(ai_response)
            
            all_concepts = list(set(query_concepts + response_concepts))
            
            # Add or update concepts
            for concept in all_concepts:
                await self._add_or_update_concept(concept, user_query + " " + ai_response)
            
            # Create relationships between concepts that appear together
            await self._create_concept_relationships(all_concepts, user_query + " " + ai_response)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge graph: {e}")
            return False
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract potential concepts from text"""
        import re
        
        # Clean and normalize text
        text = text.lower()
        
        # Technical/academic concept patterns
        concepts = []
        
        # Multi-word technical terms
        multi_word_patterns = [
            r'\b(?:machine|deep|artificial)\s+(?:learning|intelligence|network)\b',
            r'\b(?:neural|convolutional|recurrent)\s+network\b',
            r'\b(?:natural|computer)\s+(?:language|vision)\b',
            r'\b(?:data|computer)\s+(?:science|structure)\b',
            r'\b(?:software|web)\s+(?:engineering|development)\b'
        ]
        
        for pattern in multi_word_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        # Single technical terms
        single_word_concepts = [
            'algorithm', 'python', 'javascript', 'database', 'api', 'framework',
            'function', 'variable', 'array', 'object', 'class', 'method',
            'recursion', 'iteration', 'complexity', 'optimization', 'debugging',
            'programming', 'coding', 'syntax', 'semantics', 'compilation',
            'mathematics', 'calculus', 'algebra', 'statistics', 'probability',
            'physics', 'chemistry', 'biology', 'history', 'literature',
            'economics', 'psychology', 'philosophy', 'sociology'
        ]
        
        words = re.findall(r'\b[a-z]+\b', text)
        for word in words:
            if word in single_word_concepts:
                concepts.append(word)
        
        # Capitalized terms (likely proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.extend([term.lower() for term in capitalized if len(term) > 3])
        
        return list(set(concepts))[:10]  # Return unique concepts, max 10
    
    async def _add_or_update_concept(self, concept_name: str, context: str) -> str:
        """Add or update a concept in the knowledge graph"""
        concept_id = f"concept_{concept_name.replace(' ', '_')}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if concept exists
            existing = conn.execute(
                'SELECT * FROM concepts WHERE concept_id = ?', (concept_id,)
            ).fetchone()
            
            if existing:
                # Update existing concept
                new_importance = min(1.0, existing[4] + 0.1)  # Increase importance slightly
                conn.execute('''
                    UPDATE concepts 
                    SET importance_score = ?, last_accessed = julianday('now')
                    WHERE concept_id = ?
                ''', (new_importance, concept_id))
            else:
                # Create new concept
                category = self._categorize_concept(concept_name, context)
                level = self._determine_learning_level(concept_name, context)
                
                conn.execute('''
                    INSERT INTO concepts 
                    (concept_id, name, category, learning_level, related_conversations, prerequisites, dependents)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (concept_id, concept_name, category, level, '[]', '[]', '[]'))
            
            conn.commit()
            conn.close()
            
            # Update in-memory graph
            if NETWORKX_AVAILABLE and self.graph:
                if concept_id in self.graph:
                    # Update node attributes
                    self.graph.nodes[concept_id]['importance'] = new_importance if existing else 0.5
                else:
                    # Add new node
                    self.graph.add_node(concept_id, 
                                      name=concept_name,
                                      category=category if not existing else existing[3],
                                      importance=0.5,
                                      level=level if not existing else existing[5])
            
            return concept_id
            
        except Exception as e:
            self.logger.error(f"Failed to add/update concept {concept_name}: {e}")
            return ""
    
    def _categorize_concept(self, concept_name: str, context: str) -> str:
        """Determine the category of a concept"""
        concept_lower = concept_name.lower()
        context_lower = context.lower()
        
        # Programming/Computer Science
        if any(term in concept_lower or term in context_lower for term in 
               ['python', 'java', 'javascript', 'code', 'programming', 'algorithm', 'data structure']):
            return 'computer_science'
        
        # Mathematics
        if any(term in concept_lower or term in context_lower for term in 
               ['math', 'calculus', 'algebra', 'equation', 'formula', 'theorem']):
            return 'mathematics'
        
        # Science
        if any(term in concept_lower or term in context_lower for term in 
               ['physics', 'chemistry', 'biology', 'experiment', 'scientific']):
            return 'science'
        
        # Machine Learning/AI
        if any(term in concept_lower or term in context_lower for term in 
               ['machine learning', 'neural', 'artificial intelligence', 'deep learning']):
            return 'artificial_intelligence'
        
        # Default category
        return 'general'
    
    def _determine_learning_level(self, concept_name: str, context: str) -> str:
        """Determine the learning level required for a concept"""
        concept_lower = concept_name.lower()
        context_lower = context.lower()
        
        # Advanced level indicators
        if any(term in concept_lower or term in context_lower for term in 
               ['advanced', 'complex', 'sophisticated', 'theorem', 'proof', 'research']):
            return 'advanced'
        
        # Beginner level indicators
        if any(term in concept_lower or term in context_lower for term in 
               ['basic', 'introduction', 'beginner', 'simple', 'fundamental']):
            return 'beginner'
        
        # Default to intermediate
        return 'intermediate'
    
    async def _create_concept_relationships(self, concepts: List[str], context: str):
        """Create relationships between concepts that appear together"""
        if len(concepts) < 2:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create relationships between all concept pairs
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    concept1_id = f"concept_{concept1.replace(' ', '_')}"
                    concept2_id = f"concept_{concept2.replace(' ', '_')}"
                    
                    # Determine relationship type
                    rel_type = self._determine_relationship_type(concept1, concept2, context)
                    
                    # Check if relationship exists
                    existing = conn.execute('''
                        SELECT * FROM relationships 
                        WHERE (source_concept = ? AND target_concept = ?) 
                           OR (source_concept = ? AND target_concept = ?)
                    ''', (concept1_id, concept2_id, concept2_id, concept1_id)).fetchone()
                    
                    if existing:
                        # Update existing relationship
                        new_strength = min(1.0, existing[4] + 0.1)
                        new_evidence = existing[5] + 1
                        conn.execute('''
                            UPDATE relationships 
                            SET strength = ?, evidence_count = ?, last_reinforced = julianday('now')
                            WHERE id = ?
                        ''', (new_strength, new_evidence, existing[0]))
                    else:
                        # Create new relationship
                        conn.execute('''
                            INSERT INTO relationships 
                            (source_concept, target_concept, relationship_type, strength)
                            VALUES (?, ?, ?, ?)
                        ''', (concept1_id, concept2_id, rel_type, 0.5))
                        
                        # Add to in-memory graph
                        if NETWORKX_AVAILABLE and self.graph:
                            if concept1_id in self.graph.nodes and concept2_id in self.graph.nodes:
                                self.graph.add_edge(concept1_id, concept2_id, 
                                                  relationship=rel_type, strength=0.5, evidence=1)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create concept relationships: {e}")
    
    def _determine_relationship_type(self, concept1: str, concept2: str, context: str) -> str:
        """Determine the type of relationship between two concepts"""
        context_lower = context.lower()
        
        # Prerequisite relationships
        if any(word in context_lower for word in ['before', 'first', 'prerequisite', 'requires']):
            return 'prerequisite'
        
        # Part-of relationships
        if any(word in context_lower for word in ['part of', 'component', 'element', 'includes']):
            return 'part_of'
        
        # Used-in relationships
        if any(word in context_lower for word in ['used in', 'applies to', 'implementation']):
            return 'used_in'
        
        # Default to similar
        return 'similar'
    
    def get_related_concepts(self, concept_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get concepts related to the given concept"""
        concept_id = f"concept_{concept_name.replace(' ', '_')}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Get direct relationships
            relationships = conn.execute('''
                SELECT c.*, r.relationship_type, r.strength
                FROM relationships r
                JOIN concepts c ON (
                    (r.source_concept = ? AND c.concept_id = r.target_concept) OR
                    (r.target_concept = ? AND c.concept_id = r.source_concept)
                )
                WHERE r.strength > 0.3
                ORDER BY r.strength DESC, c.importance_score DESC
                LIMIT ?
            ''', (concept_id, concept_id, max_results)).fetchall()
            
            conn.close()
            
            related = []
            for rel in relationships:
                related.append({
                    'name': rel['name'],
                    'category': rel['category'],
                    'relationship_type': rel['relationship_type'],
                    'strength': rel['strength'],
                    'importance': rel['importance_score'],
                    'learning_level': rel['learning_level']
                })
            
            return related
            
        except Exception as e:
            self.logger.error(f"Failed to get related concepts: {e}")
            return []
    
    def get_learning_path(self, target_concept: str) -> List[Dict[str, Any]]:
        """Get suggested learning path to master a concept"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return []
        
        concept_id = f"concept_{target_concept.replace(' ', '_')}"
        
        if concept_id not in self.graph:
            return []
        
        try:
            # Find prerequisite path using graph traversal
            prerequisites = []
            visited = set()
            
            def find_prerequisites(node):
                if node in visited:
                    return
                visited.add(node)
                
                # Find prerequisite edges
                for pred in self.graph.predecessors(node):
                    edge_data = self.graph[pred][node]
                    if edge_data.get('relationship') == 'prerequisite':
                        node_data = self.graph.nodes[pred]
                        prerequisites.append({
                            'concept': node_data['name'],
                            'category': node_data['category'],
                            'importance': node_data['importance'],
                            'level': node_data['level']
                        })
                        find_prerequisites(pred)
            
            find_prerequisites(concept_id)
            
            # Sort prerequisites by learning level and importance
            level_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
            prerequisites.sort(key=lambda x: (level_order.get(x['level'], 1), -x['importance']))
            
            return prerequisites
            
        except Exception as e:
            self.logger.error(f"Failed to get learning path: {e}")
            return []
    
    def get_concept_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Concept statistics
            total_concepts = conn.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
            
            # Category distribution
            categories = conn.execute('''
                SELECT category, COUNT(*) as count 
                FROM concepts 
                GROUP BY category 
                ORDER BY count DESC
            ''').fetchall()
            
            # Relationship statistics
            total_relationships = conn.execute('SELECT COUNT(*) FROM relationships').fetchone()[0]
            
            # Most connected concepts
            most_connected = conn.execute('''
                SELECT c.name, COUNT(r.id) as connection_count
                FROM concepts c
                LEFT JOIN relationships r ON (c.concept_id = r.source_concept OR c.concept_id = r.target_concept)
                GROUP BY c.concept_id
                ORDER BY connection_count DESC
                LIMIT 10
            ''').fetchall()
            
            conn.close()
            
            return {
                'total_concepts': total_concepts,
                'total_relationships': total_relationships,
                'categories': [{'name': row['category'], 'count': row['count']} for row in categories],
                'most_connected': [{'name': row['name'], 'connections': row['connection_count']} for row in most_connected]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get concept statistics: {e}")
            return {}
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts by name or category"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Search by name (fuzzy matching)
            results = conn.execute('''
                SELECT * FROM concepts 
                WHERE name LIKE ? OR category LIKE ?
                ORDER BY importance_score DESC, name
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit)).fetchall()
            
            conn.close()
            
            concepts = []
            for row in results:
                concepts.append({
                    'name': row['name'],
                    'category': row['category'],
                    'importance': row['importance_score'],
                    'level': row['learning_level'],
                    'last_accessed': datetime.fromtimestamp(row['last_accessed']) if row['last_accessed'] else None
                })
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Failed to search concepts: {e}")
            return []