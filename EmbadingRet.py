from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class EmbeddingRetrieval:
    def __init__(self, db):
        self.db = db
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_exercise_embedding(self, exercise):
        """Create and store an embedding for an exercise."""
        text = f"""
        Title: {exercise['title']}
        Category: {exercise['category']}
        Instructions: {' '.join(exercise['instructions'])}
        Benefits: {' '.join(exercise['benefits'])}
        """
        
        embedding = self.model.encode(text)
        self.db.store_embedding(exercise['id'], embedding.tolist())
        return embedding

    def search_similar_exercises(self, query, n=5, filters=None):
        """
        Search for similar exercises using embeddings.
        
        Args:
            query: The search query
            n: Number of results to return
            filters: Optional dict with filters like {"category": "meditation", "difficulty": "beginner"}
        """
        query_embedding = self.model.encode(query)
        exercise_embeddings = list(self.db.db.exercise_embeddings.find({}))
        
        if not exercise_embeddings:
            return []
            
        similarities = []
        for ex_emb in exercise_embeddings:
            similarity = np.dot(query_embedding, ex_emb['embedding'])
            similarities.append((ex_emb['exercise_id'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Pre-filter to get more candidates than needed
        candidate_ids = [exercise_id for exercise_id, _ in similarities[:n*3]]
        
        # Get the actual exercises
        top_exercises = []
        for exercise_id in candidate_ids:
            exercise = self.db.get_exercise(exercise_id)
            if exercise:
                # Apply filters if any
                if filters and not self._apply_filters(exercise, filters):
                    continue
                top_exercises.append(exercise)
                if len(top_exercises) >= n:
                    break
        
        return top_exercises
    
    def _apply_filters(self, exercise, filters):
        """Apply filters to an exercise."""
        if not filters:
            return True
            
        for key, value in filters.items():
            if key not in exercise:
                return False
                
            if isinstance(value, list):
                if exercise[key] not in value:
                    return False
            elif exercise[key] != value:
                return False
                
        return True
    
    def search_by_context(self, context, n=5):
        """
        Search for exercises based on user context.
        
        Args:
            context: Dict with user context like emotional state, stress level, etc.
            n: Number of results to return
        """
        # Construct a query from the context
        query_parts = []
        
        # Add emotional state if available
        if 'emotional_state' in context:
            query_parts.append(f"Emotional state: {context['emotional_state']}")
            
        # Add needs if available
        if 'needs' in context and context['needs']:
            query_parts.append(f"Needs: {', '.join(context['needs'])}")
        
        # Add energy level as a factor
        if 'energy_level' in context:
            energy = context['energy_level']
            if energy == "low":
                query_parts.append("Low energy exercise")
            elif energy == "high":
                query_parts.append("Energetic exercise")
        
        # Add focus ability as a factor
        if 'focus_ability' in context:
            focus = context['focus_ability']
            if focus == "low":
                query_parts.append("Simple exercise requiring little concentration")
            elif focus == "high":
                query_parts.append("Exercise requiring focus and concentration")
        
        # Add stress level as a factor
        if 'stress_level' in context:
            stress = context['stress_level']
            if stress == 1 or stress is True:
                query_parts.append("Stress reduction exercise calming exercise")
        
        # Create the query
        query = " ".join(query_parts)
        if not query:
            query = "General wellness exercise"
        
        # Define filters based on context
        filters = {}
        
        # Filter by difficulty based on focus ability and energy
        if 'focus_ability' in context and context['focus_ability'] == "low":
            filters['difficulty'] = "beginner"
        elif 'energy_level' in context and context['energy_level'] == "low":
            filters['difficulty'] = ["beginner", "intermediate"]
        
        # Filter by category based on needs
        if 'needs' in context and context['needs']:
            needs = [need.lower() for need in context['needs']]
            
            # Map needs to categories
            category_mapping = {
                "relax": ["meditation", "breathing", "mindfulness"],
                "calm": ["meditation", "breathing", "mindfulness"],
                "focus": ["mindfulness", "cbt"],
                "sleep": ["meditation", "breathing"],
                "anxiety": ["breathing", "meditation", "cbt"],
                "depression": ["cbt", "gratitude", "social"],
                "stress": ["breathing", "meditation", "mindfulness"],
                "self-esteem": ["gratitude", "cbt"],
                "motivation": ["cbt", "social"]
            }
            
            categories = set()
            for need in needs:
                for key, values in category_mapping.items():
                    if key in need:
                        categories.update(values)
            
            if categories:
                filters['category'] = list(categories)
        
        return self.search_similar_exercises(query, n=n, filters=filters)