"""
Adaptive Questioning System for StudyForge AI Enhanced Edition
Implements Socratic method with adaptive difficulty and personalized questioning
"""

import random
import asyncio
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

class QuestionType(Enum):
    CLARIFICATION = "clarification"
    EXPLORATION = "exploration"
    CHALLENGE = "challenge"
    APPLICATION = "application"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"

class CognitiveDomain(Enum):
    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6

@dataclass
class AdaptiveQuestion:
    question_id: str
    text: str
    question_type: QuestionType
    cognitive_level: CognitiveDomain
    target_concept: str
    difficulty_level: float  # 0.0 to 1.0
    expected_response_indicators: List[str]
    follow_up_questions: List[str]
    learning_objective: str

@dataclass
class QuestionContext:
    user_id: str
    current_concept: str
    mastery_level: str
    confidence_score: float
    recent_confusion: List[str]
    learning_preferences: Dict[str, Any]
    session_context: Dict[str, Any]

class AdaptiveQuestioningSystem:
    """Socratic questioning system that adapts to user learning patterns"""
    
    def __init__(self, learning_tracker=None, knowledge_graph=None):
        self.learning_tracker = learning_tracker
        self.knowledge_graph = knowledge_graph
        self.db_path = "adaptive_questions.db"
        self.logger = logging.getLogger(f"{__name__}.AdaptiveQuestioningSystem")
        
        # Question templates organized by type and difficulty
        self.question_templates = self._initialize_question_templates()
        
        # User questioning patterns
        self.user_patterns = {}
        
        self._init_questions_database()
        self._load_user_patterns()
    
    def _initialize_question_templates(self) -> Dict[QuestionType, Dict[str, List[str]]]:
        """Initialize question templates for different types and difficulties"""
        return {
            QuestionType.CLARIFICATION: {
                'basic': [
                    "What do you mean when you say '{concept}'?",
                    "Can you explain '{concept}' in your own words?",
                    "How would you define '{concept}'?",
                    "What's your understanding of '{concept}'?"
                ],
                'intermediate': [
                    "How does '{concept}' relate to what you already know?",
                    "What assumptions are you making about '{concept}'?",
                    "Can you give me a concrete example of '{concept}'?",
                    "What distinguishes '{concept}' from similar ideas?"
                ],
                'advanced': [
                    "What evidence supports your understanding of '{concept}'?",
                    "How might someone with a different perspective view '{concept}'?",
                    "What are the underlying principles of '{concept}'?",
                    "How has your understanding of '{concept}' evolved?"
                ]
            },
            
            QuestionType.EXPLORATION: {
                'basic': [
                    "What happens if we change {variable} in '{concept}'?",
                    "Why do you think '{concept}' works this way?",
                    "What would happen if '{concept}' didn't exist?",
                    "How does '{concept}' connect to {related_concept}?"
                ],
                'intermediate': [
                    "What patterns do you notice in '{concept}'?",
                    "What questions does '{concept}' raise for you?",
                    "How might '{concept}' be used in different contexts?",
                    "What are the boundaries or limitations of '{concept}'?"
                ],
                'advanced': [
                    "How does '{concept}' challenge existing paradigms?",
                    "What are the implications of '{concept}' for {field}?",
                    "How might '{concept}' evolve in the future?",
                    "What alternative approaches to '{concept}' exist?"
                ]
            },
            
            QuestionType.APPLICATION: {
                'basic': [
                    "Where might you use '{concept}' in real life?",
                    "Can you think of a situation where '{concept}' applies?",
                    "How would you explain '{concept}' to a friend?",
                    "What problems does '{concept}' help solve?"
                ],
                'intermediate': [
                    "How would you apply '{concept}' to solve {problem}?",
                    "What steps would you take to implement '{concept}'?",
                    "How would you modify '{concept}' for {scenario}?",
                    "What tools or resources would you need to use '{concept}'?"
                ],
                'advanced': [
                    "How would you optimize '{concept}' for {constraint}?",
                    "What trade-offs exist when applying '{concept}'?",
                    "How would you scale '{concept}' to larger problems?",
                    "What ethical considerations arise from using '{concept}'?"
                ]
            },
            
            QuestionType.CHALLENGE: {
                'basic': [
                    "What if someone disagreed with '{concept}'? What would you say?",
                    "Can you think of any exceptions to '{concept}'?",
                    "What's the weakest part of '{concept}'?",
                    "How confident are you about '{concept}' and why?"
                ],
                'intermediate': [
                    "What evidence might contradict '{concept}'?",
                    "How would you respond to criticism of '{concept}'?",
                    "What assumptions does '{concept}' make that might be wrong?",
                    "How does '{concept}' handle edge cases or exceptions?"
                ],
                'advanced': [
                    "How would you test the validity of '{concept}'?",
                    "What are the philosophical implications of '{concept}'?",
                    "How does '{concept}' interact with conflicting theories?",
                    "What would convince you that '{concept}' is wrong?"
                ]
            },
            
            QuestionType.SYNTHESIS: {
                'basic': [
                    "How does '{concept}' fit with {other_concept}?",
                    "What do '{concept}' and {other_concept} have in common?",
                    "Can you combine '{concept}' with something else?",
                    "How would you summarize '{concept}' in one sentence?"
                ],
                'intermediate': [
                    "How would you integrate '{concept}' with {system}?",
                    "What patterns connect '{concept}' to {domain}?",
                    "How does understanding '{concept}' change your view of {field}?",
                    "What framework could unite '{concept}' with {other_concepts}?"
                ],
                'advanced': [
                    "How would you design a system that incorporates '{concept}'?",
                    "What new insights emerge from combining '{concept}' with {theory}?",
                    "How could '{concept}' revolutionize {field}?",
                    "What original applications of '{concept}' can you envision?"
                ]
            },
            
            QuestionType.EVALUATION: {
                'basic': [
                    "Do you think '{concept}' is important? Why?",
                    "What's good and bad about '{concept}'?",
                    "Would you recommend '{concept}' to others?",
                    "How useful is '{concept}' for learning?"
                ],
                'intermediate': [
                    "How would you judge the effectiveness of '{concept}'?",
                    "What criteria would you use to evaluate '{concept}'?",
                    "How does '{concept}' compare to alternatives?",
                    "What impact has '{concept}' had on {field}?"
                ],
                'advanced': [
                    "What are the long-term consequences of '{concept}'?",
                    "How would you assess the ethical implications of '{concept}'?",
                    "What standards should we use to judge '{concept}'?",
                    "How has '{concept}' influenced the development of {domain}?"
                ]
            }
        }
    
    def _init_questions_database(self):
        """Initialize adaptive questioning database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Generated questions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS generated_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                cognitive_level INTEGER NOT NULL,
                target_concept TEXT NOT NULL,
                difficulty_level REAL NOT NULL,
                context_data TEXT,
                generated_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Question responses table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS question_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                response_text TEXT,
                response_quality REAL,
                thinking_time_seconds REAL,
                follow_up_needed BOOLEAN DEFAULT FALSE,
                mastery_indicators TEXT,
                confusion_indicators TEXT,
                timestamp REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # User questioning preferences
        conn.execute('''
            CREATE TABLE IF NOT EXISTS questioning_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                preferred_question_types TEXT,
                difficulty_preference REAL DEFAULT 0.5,
                cognitive_level_preference INTEGER DEFAULT 3,
                response_style TEXT DEFAULT 'adaptive',
                last_updated REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_questions_user ON generated_questions(user_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_responses_question ON question_responses(question_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_user ON questioning_preferences(user_id)')
        
        conn.commit()
        conn.close()
    
    def _load_user_patterns(self):
        """Load user questioning patterns from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            users = conn.execute('SELECT * FROM questioning_preferences').fetchall()
            
            for user in users:
                self.user_patterns[user['user_id']] = {
                    'preferred_types': json.loads(user['preferred_question_types'] or '[]'),
                    'difficulty_preference': user['difficulty_preference'],
                    'cognitive_preference': user['cognitive_level_preference'],
                    'response_style': user['response_style']
                }
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to load user patterns: {e}")
    
    async def should_ask_questions(self, user_query: str, user_id: str) -> List[str]:
        """Determine if questions should be asked and generate them"""
        try:
            # Analyze user query for questioning opportunities
            context = await self._analyze_query_context(user_query, user_id)
            
            # Check if questioning would be beneficial
            if not self._should_question(context):
                return []
            
            # Generate adaptive questions
            questions = await self._generate_adaptive_questions(context)
            
            return [q.text for q in questions]
            
        except Exception as e:
            self.logger.error(f"Failed to determine questioning need: {e}")
            return []
    
    async def _analyze_query_context(self, query: str, user_id: str) -> QuestionContext:
        """Analyze the context of the user's query"""
        
        # Extract concepts from query
        concepts = self._extract_concepts_from_query(query)
        primary_concept = concepts[0] if concepts else "general"
        
        # Get user's mastery level for primary concept
        mastery_level = "intermediate"  # Default
        confidence_score = 0.5
        
        if self.learning_tracker:
            try:
                masteries = self.learning_tracker.get_concept_mastery(user_id, primary_concept)
                if masteries:
                    mastery = masteries[0]
                    mastery_level = mastery.mastery_level.name.lower()
                    confidence_score = mastery.confidence_score
            except:
                pass
        
        # Analyze for confusion indicators
        confusion_indicators = []
        confusion_words = ['confused', 'unclear', 'don\'t understand', 'help', 'explain', 'what is']
        for word in confusion_words:
            if word in query.lower():
                confusion_indicators.append(word)
        
        # Get user preferences
        preferences = self.user_patterns.get(user_id, {
            'preferred_types': [],
            'difficulty_preference': 0.5,
            'cognitive_preference': 3,
            'response_style': 'adaptive'
        })
        
        return QuestionContext(
            user_id=user_id,
            current_concept=primary_concept,
            mastery_level=mastery_level,
            confidence_score=confidence_score,
            recent_confusion=confusion_indicators,
            learning_preferences=preferences,
            session_context={'query': query, 'concepts': concepts}
        )
    
    def _should_question(self, context: QuestionContext) -> bool:
        """Determine if questioning would be beneficial"""
        
        # Always question if user shows confusion
        if context.recent_confusion:
            return True
        
        # Question if confidence is low
        if context.confidence_score < 0.4:
            return True
        
        # Question if mastery level suggests room for deeper understanding
        if context.mastery_level in ['beginner', 'learning']:
            return random.random() < 0.7  # 70% chance
        elif context.mastery_level == 'practicing':
            return random.random() < 0.4  # 40% chance
        elif context.mastery_level in ['proficient', 'expert']:
            return random.random() < 0.2  # 20% chance for advanced questions
        
        return False
    
    async def _generate_adaptive_questions(self, context: QuestionContext) -> List[AdaptiveQuestion]:
        """Generate questions adapted to user's context"""
        questions = []
        
        try:
            # Determine question types to use
            question_types = self._select_question_types(context)
            
            for question_type in question_types[:2]:  # Generate up to 2 questions
                question = self._create_question(question_type, context)
                if question:
                    questions.append(question)
                    
                    # Store generated question
                    await self._store_generated_question(question, context)
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptive questions: {e}")
        
        return questions
    
    def _select_question_types(self, context: QuestionContext) -> List[QuestionType]:
        """Select appropriate question types based on context"""
        
        # If user shows confusion, start with clarification
        if context.recent_confusion:
            return [QuestionType.CLARIFICATION]
        
        # Based on mastery level
        if context.mastery_level in ['beginner', 'learning']:
            return [QuestionType.CLARIFICATION, QuestionType.EXPLORATION]
        elif context.mastery_level == 'practicing':
            return [QuestionType.APPLICATION, QuestionType.EXPLORATION]
        elif context.mastery_level == 'proficient':
            return [QuestionType.CHALLENGE, QuestionType.SYNTHESIS]
        elif context.mastery_level == 'expert':
            return [QuestionType.EVALUATION, QuestionType.SYNTHESIS]
        
        return [QuestionType.EXPLORATION]
    
    def _create_question(self, question_type: QuestionType, context: QuestionContext) -> Optional[AdaptiveQuestion]:
        """Create a specific question based on type and context"""
        
        # Determine difficulty level
        difficulty_level = self._calculate_question_difficulty(context)
        
        # Select difficulty category
        if difficulty_level < 0.4:
            difficulty_cat = 'basic'
        elif difficulty_level < 0.7:
            difficulty_cat = 'intermediate'
        else:
            difficulty_cat = 'advanced'
        
        # Get question templates for this type and difficulty
        templates = self.question_templates.get(question_type, {}).get(difficulty_cat, [])
        
        if not templates:
            return None
        
        # Select a random template
        template = random.choice(templates)
        
        # Fill in the template
        question_text = self._fill_question_template(template, context)
        
        # Determine cognitive level
        cognitive_level = self._determine_cognitive_level(question_type, difficulty_level)
        
        # Generate question ID
        question_id = f"q_{context.user_id}_{int(datetime.now().timestamp())}"
        
        # Create follow-up questions
        follow_ups = self._generate_follow_up_questions(question_type, context)
        
        return AdaptiveQuestion(
            question_id=question_id,
            text=question_text,
            question_type=question_type,
            cognitive_level=cognitive_level,
            target_concept=context.current_concept,
            difficulty_level=difficulty_level,
            expected_response_indicators=[],
            follow_up_questions=follow_ups,
            learning_objective=f"Deepen understanding of {context.current_concept}"
        )
    
    def _calculate_question_difficulty(self, context: QuestionContext) -> float:
        """Calculate appropriate question difficulty"""
        base_difficulty = 0.5
        
        # Adjust based on mastery level
        mastery_adjustments = {
            'beginner': -0.2,
            'learning': -0.1,
            'practicing': 0.0,
            'proficient': 0.1,
            'expert': 0.2
        }
        
        base_difficulty += mastery_adjustments.get(context.mastery_level, 0.0)
        
        # Adjust based on confidence
        if context.confidence_score < 0.3:
            base_difficulty -= 0.1
        elif context.confidence_score > 0.8:
            base_difficulty += 0.1
        
        # User preference adjustment
        pref_difficulty = context.learning_preferences.get('difficulty_preference', 0.5)
        base_difficulty = (base_difficulty + pref_difficulty) / 2
        
        return max(0.1, min(0.9, base_difficulty))
    
    def _fill_question_template(self, template: str, context: QuestionContext) -> str:
        """Fill in question template with context-specific information"""
        
        # Replace placeholders
        question = template.replace('{concept}', context.current_concept)
        
        # Add related concepts if available
        if self.knowledge_graph:
            related_concepts = self.knowledge_graph.get_related_concepts(context.current_concept, 3)
            if related_concepts:
                related_concept = related_concepts[0]['name']
                question = question.replace('{related_concept}', related_concept)
                question = question.replace('{other_concept}', related_concept)
        
        # Replace other placeholders with generic terms
        replacements = {
            '{variable}': 'a parameter',
            '{problem}': 'a specific problem',
            '{scenario}': 'a different scenario',
            '{field}': 'this field',
            '{domain}': 'this domain',
            '{system}': 'a larger system',
            '{theory}': 'related theory',
            '{constraint}': 'specific constraints'
        }
        
        for placeholder, replacement in replacements.items():
            question = question.replace(placeholder, replacement)
        
        return question
    
    def _determine_cognitive_level(self, question_type: QuestionType, difficulty: float) -> CognitiveDomain:
        """Determine cognitive domain level for the question"""
        
        type_mapping = {
            QuestionType.CLARIFICATION: CognitiveDomain.UNDERSTAND,
            QuestionType.EXPLORATION: CognitiveDomain.APPLY,
            QuestionType.APPLICATION: CognitiveDomain.APPLY,
            QuestionType.CHALLENGE: CognitiveDomain.ANALYZE,
            QuestionType.SYNTHESIS: CognitiveDomain.CREATE,
            QuestionType.EVALUATION: CognitiveDomain.EVALUATE
        }
        
        base_level = type_mapping.get(question_type, CognitiveDomain.UNDERSTAND)
        
        # Adjust based on difficulty
        if difficulty > 0.7:
            # Increase cognitive level for high difficulty
            current_level = base_level.value
            higher_level = min(6, current_level + 1)
            return CognitiveDomain(higher_level)
        elif difficulty < 0.3:
            # Decrease cognitive level for low difficulty
            current_level = base_level.value
            lower_level = max(1, current_level - 1)
            return CognitiveDomain(lower_level)
        
        return base_level
    
    def _generate_follow_up_questions(self, question_type: QuestionType, context: QuestionContext) -> List[str]:
        """Generate follow-up questions based on the main question"""
        
        follow_ups = []
        concept = context.current_concept
        
        if question_type == QuestionType.CLARIFICATION:
            follow_ups = [
                f"Can you give me an example of {concept}?",
                f"How does {concept} work in practice?",
                f"What makes {concept} different from similar concepts?"
            ]
        elif question_type == QuestionType.EXPLORATION:
            follow_ups = [
                f"What other applications of {concept} can you think of?",
                f"How might {concept} be improved or modified?",
                f"What questions does this raise about {concept}?"
            ]
        elif question_type == QuestionType.APPLICATION:
            follow_ups = [
                f"What challenges might you face when using {concept}?",
                f"How would you know if you're applying {concept} correctly?",
                f"What resources would help you implement {concept}?"
            ]
        elif question_type == QuestionType.CHALLENGE:
            follow_ups = [
                f"How would you defend {concept} against criticism?",
                f"What evidence supports the validity of {concept}?",
                f"How has {concept} been tested or validated?"
            ]
        
        return follow_ups[:2]  # Return top 2 follow-ups
    
    async def _store_generated_question(self, question: AdaptiveQuestion, context: QuestionContext):
        """Store generated question in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO generated_questions 
                (question_id, user_id, question_text, question_type, cognitive_level,
                 target_concept, difficulty_level, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                question.question_id, context.user_id, question.text,
                question.question_type.value, question.cognitive_level.value,
                question.target_concept, question.difficulty_level,
                json.dumps(context.session_context)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store generated question: {e}")
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract concepts from user query - simplified version"""
        import re
        
        # Simple concept extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        
        # Filter for likely concepts
        concept_words = [
            word for word in set(words)
            if word not in ['what', 'how', 'why', 'when', 'where', 'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been']
        ]
        
        return concept_words[:5]  # Return top 5 concepts
    
    async def record_question_response(self, question_id: str, user_id: str, 
                                     response_text: str, response_time: float) -> bool:
        """Record user's response to a question"""
        try:
            # Analyze response quality (simplified)
            response_quality = self._analyze_response_quality(response_text)
            
            # Store response
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO question_responses 
                (question_id, user_id, response_text, response_quality, thinking_time_seconds)
                VALUES (?, ?, ?, ?, ?)
            ''', (question_id, user_id, response_text, response_quality, response_time))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record question response: {e}")
            return False
    
    def _analyze_response_quality(self, response: str) -> float:
        """Analyze the quality of a user's response"""
        if not response or len(response.strip()) < 10:
            return 0.2
        
        # Simple quality indicators
        quality_score = 0.5
        
        # Length bonus (up to reasonable limit)
        word_count = len(response.split())
        if word_count > 20:
            quality_score += 0.2
        elif word_count > 50:
            quality_score += 0.3
        
        # Specific examples or details
        if any(phrase in response.lower() for phrase in ['for example', 'such as', 'specifically', 'in particular']):
            quality_score += 0.1
        
        # Reasoning indicators
        if any(phrase in response.lower() for phrase in ['because', 'therefore', 'however', 'although', 'since']):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def get_questioning_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics about user's questioning patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Question statistics
            total_questions = conn.execute(
                'SELECT COUNT(*) FROM generated_questions WHERE user_id = ?', (user_id,)
            ).fetchone()[0]
            
            # Response statistics
            response_stats = conn.execute('''
                SELECT AVG(response_quality) as avg_quality, 
                       AVG(thinking_time_seconds) as avg_thinking_time,
                       COUNT(*) as response_count
                FROM question_responses 
                WHERE user_id = ?
            ''', (user_id,)).fetchone()
            
            # Question type distribution
            type_dist = conn.execute('''
                SELECT question_type, COUNT(*) as count
                FROM generated_questions 
                WHERE user_id = ?
                GROUP BY question_type
            ''', (user_id,)).fetchall()
            
            conn.close()
            
            return {
                'total_questions_generated': total_questions,
                'total_responses': response_stats['response_count'] if response_stats else 0,
                'average_response_quality': response_stats['avg_quality'] if response_stats else 0,
                'average_thinking_time': response_stats['avg_thinking_time'] if response_stats else 0,
                'question_type_distribution': [
                    {'type': row['question_type'], 'count': row['count']} 
                    for row in type_dist
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get questioning analytics: {e}")
            return {}