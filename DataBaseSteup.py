# File: database_setup.py
from pymongo import MongoClient
from datetime import datetime

class Database:
    def __init__(self, uri="mongodb://localhost:27017"):
        self.client = MongoClient(uri)
        self.db = self.client.mental_wellness

    def setup_collections(self):
        try:
            # Create exercises collection with validation
            if "exercises" not in self.db.list_collection_names():
                self.db.create_collection("exercises", 
                    validator={
                        '$jsonSchema': {
                            'bsonType': "object",
                            'required': ["id", "title", "category", "difficulty", "duration", "instructions", "benefits"],
                            'properties': {
                                'id': { 'bsonType': "string" },
                                'title': { 'bsonType': "string" },
                                'category': { 
                                    'enum': ["meditation", "breathing", "journaling", "social", 
                                            "mindfulness", "emotional_regulation", "cbt", "gratitude"] 
                                },
                                'difficulty': { 
                                    'enum': ["beginner", "intermediate", "advanced"] 
                                },
                                'duration': { 'bsonType': "int" },
                                'instructions': {
                                    'bsonType': "array",
                                    'items': { 'bsonType': "string" }
                                },
                                'benefits': {
                                    'bsonType': "array",
                                    'items': { 'bsonType': "string" }
                                }
                            }
                        }
                    }
                )

            # Create indexes
            self.db.exercises.create_index([("category", 1), ("difficulty", 1)])
            self.db.exercises.create_index([("duration", 1)])

            # Create other collections
            if "exercise_embeddings" not in self.db.list_collection_names():
                self.db.create_collection("exercise_embeddings")
                self.db.exercise_embeddings.create_index([("exercise_id", 1)])

            if "users" not in self.db.list_collection_names():
                self.db.create_collection("users")
                self.db.users.create_index([("user_id", 1)], unique=True)
                
            # Nouvelle collection pour les exercices personnalisés
            if "personalized_exercises" not in self.db.list_collection_names():
                self.db.create_collection("personalized_exercises")
                self.db.personalized_exercises.create_index([("user_id", 1)])
                self.db.personalized_exercises.create_index([("source_id", 1)])

            print("Collections set up successfully")
        except Exception as e:
            print(f"Error setting up collections: {e}")
            raise

    def insert_exercise(self, exercise_data):
        """Insert a new exercise into the database."""
        try:
            # Add timestamp
            exercise_data['created_at'] = datetime.utcnow()
            
            # Ensure required fields
            required_fields = ["id", "title", "category", "difficulty", "duration", "instructions", "benefits"]
            for field in required_fields:
                if field not in exercise_data:
                    raise ValueError(f"Missing required field: {field}")

            # Insert the exercise
            result = self.db.exercises.insert_one(exercise_data)
            
            print(f"Exercise inserted successfully with ID: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            print(f"Error inserting exercise: {e}")
            raise

    def get_exercise(self, exercise_id):
        """Retrieve an exercise by ID."""
        try:
            return self.db.exercises.find_one({"id": exercise_id})
        except Exception as e:
            print(f"Error retrieving exercise: {e}")
            return None

    def get_all_exercises(self):
        """Retrieve all exercises."""
        try:
            return list(self.db.exercises.find({}))
        except Exception as e:
            print(f"Error retrieving exercises: {e}")
            return []

    def store_embedding(self, exercise_id, embedding):
        """Store embedding for an exercise."""
        try:
            return self.db.exercise_embeddings.update_one(
                {"exercise_id": exercise_id},
                {
                    "$set": {
                        "embedding": embedding,
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"Error storing embedding: {e}")
            raise

    def update_exercise(self, exercise_id, update_data):
        """Update an existing exercise."""
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = self.db.exercises.update_one(
                {"id": exercise_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating exercise: {e}")
            return False

    def delete_exercise(self, exercise_id):
        """Delete an exercise and its embedding."""
        try:
            # Delete exercise
            ex_result = self.db.exercises.delete_one({"id": exercise_id})
            # Delete embedding
            emb_result = self.db.exercise_embeddings.delete_one({"exercise_id": exercise_id})
            return ex_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting exercise: {e}")
            return False
            
    # Nouvelles méthodes pour les exercices personnalisés
    def store_personalized_exercise(self, exercise_data, user_id):
        """Store a personalized exercise for a user."""
        try:
            # Ensure personalized flag is set
            exercise_data['personalized'] = True
            exercise_data['user_id'] = user_id
            exercise_data['created_at'] = datetime.utcnow()
            
            # Add source_id if not present
            if 'source_id' not in exercise_data and 'id' in exercise_data:
                exercise_data['source_id'] = exercise_data['id']
                
            # Generate a new ID for the personalized exercise
            if 'id' in exercise_data:
                exercise_data['personalized_id'] = f"PERS_{exercise_data['id']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            else:
                exercise_data['personalized_id'] = f"PERS_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            result = self.db.personalized_exercises.insert_one(exercise_data)
            print(f"Personalized exercise stored with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"Error storing personalized exercise: {e}")
            raise
            
    def get_user_personalized_exercises(self, user_id):
        """Get personalized exercises for a specific user."""
        try:
            return list(self.db.personalized_exercises.find({"user_id": user_id}))
        except Exception as e:
            print(f"Error retrieving personalized exercises: {e}")
            return []

    def close(self):
        """Close the database connection."""
        try:
            self.client.close()
            print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")