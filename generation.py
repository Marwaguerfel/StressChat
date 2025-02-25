# File: exercise_generation.py
from openai import OpenAI
from datetime import datetime
import json
from key import OPENAI_API_KEY
import asyncio
from groq import Groq
from key import GROQ_API_KEY

class ExerciseGenerator:
    def __init__(self, db, embedding_retrieval):
        self.db = db
        self.embedding_retrieval = embedding_retrieval
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.valid_categories = ["meditation", "breathing", "journaling", "social", 
                               "mindfulness", "emotional_regulation", "cbt", "gratitude"]
        self.valid_difficulties = ["beginner", "intermediate", "advanced"]

    async def generate_exercise(self, query, user_id=None):
        """Generate an exercise using traditional method."""
        try:
            print(f"Generating exercise for query: {query}")
            similar_exercises = self.embedding_retrieval.search_similar_exercises(query)
            print(f"Found {len(similar_exercises)} similar exercises")
            
            # Create the context and prompt (no changes needed)
            context = {
                'query': query,
                'similar_count': len(similar_exercises)
            }

            prompt = f"""Create a mental wellness exercise with these requirements:
            Query: {query}
            
            The exercise must include:
            - A clear title
            - Category (one of: {', '.join(self.valid_categories)})
            - Difficulty (one of: {', '.join(self.valid_difficulties)})
            - Duration (in seconds)
            - A list of instructions
            - A list of benefits
            - Relevant tags

            Return the exercise in this exact JSON format:
            {{
                "title": "Exercise Title",
                "category": "category_name",
                "difficulty": "difficulty_level",
                "duration": seconds_number,
                "instructions": ["step 1", "step 2", "step 3"],
                "benefits": ["benefit 1", "benefit 2"],
                "tags": ["tag1", "tag2"]
            }}"""

            print("Sending request to OpenAI...")
            # Run OpenAI request in the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a mental wellness exercise creator. Create clear, practical exercises in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
            )
            
            print("Received response from OpenAI")
            response_content = response.choices[0].message.content
            print(f"Raw response: {response_content}")

            new_exercise = self._process_response(response_content)
            print(f"Processed exercise: {json.dumps(new_exercise, indent=2)}")
            
            if self._validate_exercise(new_exercise):
                print("Exercise validated successfully")
                # Run database operations in the event loop
                await loop.run_in_executor(None, self.db.insert_exercise, new_exercise)
                await loop.run_in_executor(None, self.embedding_retrieval.create_exercise_embedding, new_exercise)
                return new_exercise
            else:
                print("Exercise validation failed")
                raise ValueError("Generated exercise failed validation")
            
        except Exception as e:
            print(f"Error in generate_exercise: {str(e)}")
            return None

    def _process_response(self, content):
        try:
            print("Processing response...")
            # Clean the response string
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            print(f"Cleaned content: {content}")

            # Parse JSON response
            exercise_data = json.loads(content)
            
            # Add generated ID
            exercise_data['id'] = f"EX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Ensure all required fields exist
            required_fields = ['title', 'category', 'difficulty', 'duration', 
                             'instructions', 'benefits', 'tags']
            for field in required_fields:
                if field not in exercise_data:
                    raise ValueError(f"Missing required field: {field}")

            return exercise_data

        except Exception as e:
            print(f"Error processing response: {str(e)}")
            # Return a valid default exercise
            return {
                'id': f"EX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'title': "Basic Mindfulness Exercise",
                'category': "mindfulness",
                'difficulty': "beginner",
                'duration': 300,
                'instructions': ["Focus on your breath", "Notice your thoughts", "Return to breath"],
                'benefits': ["Reduces stress", "Improves focus"],
                'tags': ["mindfulness", "beginner-friendly", "stress-relief"]
            }

    def _validate_exercise(self, exercise):
        print("Validating exercise...")
        try:
            # Validate category
            if exercise['category'] not in self.valid_categories:
                print(f"Invalid category: {exercise['category']}")
                return False

            # Validate difficulty
            if exercise['difficulty'] not in self.valid_difficulties:
                print(f"Invalid difficulty: {exercise['difficulty']}")
                return False

            # Validate duration
            if not isinstance(exercise['duration'], int) or exercise['duration'] <= 0:
                print(f"Invalid duration: {exercise['duration']}")
                return False

            # Validate lists
            if not isinstance(exercise['instructions'], list) or not exercise['instructions']:
                print("Invalid or empty instructions")
                return False

            if not isinstance(exercise['benefits'], list) or not exercise['benefits']:
                print("Invalid or empty benefits")
                return False

            print("Exercise validated successfully")
            return True

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False
    
    # Nouvelles méthodes pour la génération d'exercices avec RAG
    
    async def analyze_conversation(self, conversation_text, stress_level):
        """Analyze conversation to extract context for exercise generation."""
        try:
            stress_context = "high stress levels" if stress_level == 1 else "normal stress levels"
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mental health analyst. Extract key information from this conversation."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this conversation between a user and a mental health assistant. The user shows {stress_context}.\n\n{conversation_text}\n\nExtract the following information in JSON format:\n- emotional_state: description of user's emotional state\n- interests: list of potential interests or preferences mentioned\n- needs: list of psychological or emotional needs identified\n- energy_level: estimate of energy level (low, medium, high)\n- focus_ability: estimate of ability to focus (low, medium, high)"
                    }
                ],
                model="qwen-2.5-32b",
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing conversation: {e}")
            # Valeurs par défaut en cas d'échec
            return {
                "emotional_state": "neutral",
                "interests": [],
                "needs": ["relaxation"] if stress_level == 1 else ["wellbeing"],
                "energy_level": "low" if stress_level == 1 else "medium",
                "focus_ability": "low" if stress_level == 1 else "medium"
            }
    
    async def generate_exercise_from_conversation(self, conversation, stress_level=0, user_id='default'):
        """Generate a personalized exercise based on conversation analysis using RAG."""
        try:
            # Format conversation for analysis
            conversation_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['message']}"
                for msg in conversation if 'message' in msg
            ])
            
            # Analyze conversation
            analysis = await self.analyze_conversation(conversation_text, stress_level)
            print(f"Conversation analysis: {json.dumps(analysis)}")
            
            # Find similar exercises based on context
            similar_exercises = self.embedding_retrieval.search_by_context(
                context={**analysis, 'stress_level': stress_level},
                n=3
            )
            
            if not similar_exercises:
                print("No similar exercises found, falling back to standard generation")
                return await self.generate_exercise(conversation_text, user_id)
            
            print(f"Found {len(similar_exercises)} similar exercises for personalization")
            
            # Format exercises for prompt
            exercises_text = ""
            for i, exercise in enumerate(similar_exercises):
                exercises_text += f"""Exercise {i+1}:
Title: {exercise.get('title', '')}
Category: {exercise.get('category', '')}
Difficulty: {exercise.get('difficulty', '')}
Duration: {exercise.get('duration', 0)} seconds
Instructions: {' '.join(exercise.get('instructions', []))}
Benefits: {' '.join(exercise.get('benefits', []))}

"""
            
            # Personalize an exercise based on the analysis
            prompt = f"""
            As a mental health expert, create a personalized wellness exercise for this user.
            
            USER ANALYSIS:
            - Emotional state: {analysis.get('emotional_state', 'neutral')}
            - Energy level: {analysis.get('energy_level', 'medium')}
            - Focus ability: {analysis.get('focus_ability', 'medium')}
            - Stress level: {"high" if stress_level == 1 else "normal"}
            - Needs: {', '.join(analysis.get('needs', []))}
            
            REFERENCE EXERCISES:
            {exercises_text}
            
            Create a personalized exercise for this user that addresses their specific needs.
            Follow these requirements:
            - Category must be one of: {', '.join(self.valid_categories)}
            - Difficulty must be one of: {', '.join(self.valid_difficulties)}
            - Ensure the difficulty level is appropriate for the user's energy and focus
            - Duration should be appropriate for the exercise type (in seconds)
            - Instructions should be clear and detailed
            - Include specific benefits that address the user's needs
            
            Return the exercise in this exact JSON format:
            {{
                "title": "Exercise Title",
                "category": "category_name",
                "difficulty": "difficulty_level",
                "duration": seconds_number,
                "instructions": ["step 1", "step 2", "step 3"],
                "benefits": ["benefit 1", "benefit 2"],
                "tags": ["tag1", "tag2"]
            }}
            """
            
            # Get personalized exercise from Groq
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a mental health expert specializing in personalized wellness exercises."},
                    {"role": "user", "content": prompt}
                ],
                model="qwen-2.5-32b",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            personalized_data = json.loads(response.choices[0].message.content)
            
            # Add ID and metadata
            personalized_data['id'] = f"EX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            personalized_data['source_id'] = similar_exercises[0]['id']
            personalized_data['personalized'] = True
            
            # Validate the exercise
            if self._validate_exercise(personalized_data):
                print("Personalized exercise validated successfully")
                
                # Store in database
                await loop.run_in_executor(None, self.db.insert_exercise, personalized_data)
                await loop.run_in_executor(None, self.embedding_retrieval.create_exercise_embedding, personalized_data)
                
                # Also store as a personalized exercise
                await loop.run_in_executor(None, self.db.store_personalized_exercise, personalized_data, user_id)
                
                return personalized_data
            else:
                print("Personalized exercise validation failed, returning reference exercise")
                # Return the first similar exercise as fallback
                return similar_exercises[0]
                
        except Exception as e:
            print(f"Error in generate_exercise_from_conversation: {str(e)}")
            
            # Fallback to the standard generator if RAG fails
            try:
                return await self.generate_exercise("wellness exercise for stress management", user_id)
            except:
                # Ultimate fallback - return a default exercise
                return {
                    'id': f"EX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'title': "Basic Mindfulness Exercise",
                    'category': "mindfulness",
                    'difficulty': "beginner",
                    'duration': 300,
                    'instructions': ["Focus on your breath", "Notice your thoughts", "Return to breath"],
                    'benefits': ["Reduces stress", "Improves focus"],
                    'tags': ["mindfulness", "beginner-friendly", "stress-relief"]
                }