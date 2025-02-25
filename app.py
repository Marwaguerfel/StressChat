from flask import Flask, render_template, request, jsonify, make_response
from groq import Groq
import torch
from model.predictor import StressPredictor
from DataBaseSteup import Database  # Note: Should be "DatabaseSetup" (typo fix)
from EmbadingRet import EmbeddingRetrieval  # Note: Should be "EmbeddingRetrieval" (typo fix)
from generation import ExerciseGenerator
import os
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from key import GROQ_API_KEY
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import asyncio
from bson import ObjectId

app = Flask(__name__, static_folder="static")
app.secret_key = os.urandom(24)

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = MongoJSONEncoder

# Initialize systems
db = Database()
db.setup_collections()
embedding_retrieval = EmbeddingRetrieval(db)
generator = ExerciseGenerator(db, embedding_retrieval)
groq_client = Groq(api_key=GROQ_API_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stress_predictor = StressPredictor(device)
conversations = {}

def setup_logging():
    """Configure application logging"""
    if not app.debug:
        os.makedirs('logs', exist_ok=True)
        file_handler = RotatingFileHandler(
            'logs/chatbot.log',
            maxBytes=10240,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

# Utility functions
def get_stress_prompt(stress_level):
    """Return system prompt based on stress level"""
    base_prompt = "You are an empathetic mental health support assistant"
    return (
        f"{base_prompt} with expertise in stress management..."
        if stress_level == 1
        else f"{base_prompt} focused on positive mental wellbeing..."
    )

def get_groq_response(user_message, stress_level, confidence, user_id='default'):
    """Get response from Groq API with appropriate context"""
    try:
        history = conversations.get(user_id, [])
        messages = [{"role": "system", "content": get_stress_prompt(stress_level)}]
        messages.extend([
            {"role": msg["role"], "content": msg["message"]} 
            for msg in history[-5:]
        ])
        messages.extend([
            {
                "role": "system",
                "content": f"[User stress analysis: level {stress_level}, confidence {confidence:.2f}]"
            },
            {"role": "user", "content": user_message}
        ])

        completion = groq_client.chat.completions.create(
            messages=messages,
            model="qwen-2.5-32b",  # Verify this model exists
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        
        response = completion.choices[0].message.content
        
        if user_id not in conversations:
            conversations[user_id] = []
            
        conversations[user_id].extend([
            {
                "timestamp": datetime.now().isoformat(),
                "message": user_message,
                "role": "user",
                "stress_analysis": {"level": stress_level, "confidence": confidence}
            },
            {
                "timestamp": datetime.now().isoformat(),
                "message": response,
                "role": "assistant"
            }
        ])
        
        return response
        
    except Exception as e:
        app.logger.error(f"Groq API error: {str(e)}")
        return "I apologize, but I'm having trouble connecting right now. Please try again."

def generate_summary(conversation):
    """Generate conversation summary using Groq"""
    try:
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['message']}"
            for msg in conversation
        ])

        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a mental health expert. Provide concise summaries (3-4 sentences) focusing on critical points."
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation briefly, covering main topic, emotional state, and critical issues:\n\n{conversation_text}"
                }
            ],
            model="qwen-2.5-32b",
            temperature=0.5,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Summary generation error: {str(e)}")
        return "Unable to generate summary at this time."

# Routes
@app.route('/')
def home():
    try:
        exercises = db.get_all_exercises()
        return render_template('index.html', exercises=exercises)
    except Exception as e:
        app.logger.error(f"Home route error: {str(e)}")
        return render_template('index.html', exercises=[])

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message'].strip()
        user_id = data.get('user_id', 'default')

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        stress_result = stress_predictor.predict(user_message)
        response = get_groq_response(
            user_message,
            stress_result['stress_level'],
            stress_result['confidence'],
            user_id
        )

        return jsonify({
            'message': user_message,
            'response': response,
            'stress_analysis': stress_result
        })

    except Exception as e:
        app.logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate-exercise', methods=['POST'])
async def generate_exercise_from_summary():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'error': 'User ID required'}), 400
            
        user_id = data['user_id']
        
        if user_id not in conversations:
            return jsonify({'error': 'No conversation found'}), 404
            
        conversation = conversations[user_id]
        recent_messages = [msg for msg in conversation[-5:] 
                         if msg['role'] == 'user' and 'stress_analysis' in msg]
        
        stress_level = (
            1 if recent_messages and 
            sum(msg['stress_analysis']['level'] for msg in recent_messages) / len(recent_messages) > 0.5
            else 0
        )
        
        app.logger.info(f"Generating exercise with RAG method, stress level: {stress_level}")
        
        new_exercise = await generator.generate_exercise_from_conversation(
            conversation, 
            stress_level,
            user_id
        )
        
        new_exercise = json.loads(json.dumps(new_exercise, cls=MongoJSONEncoder))
        
        app.logger.info(f"Generated RAG exercise: {new_exercise.get('title', 'Unknown')}")
        
        return jsonify({
            'success': True,
            'exercise': new_exercise,
            'method': 'rag'
        })

    except Exception as e:
        app.logger.error(f"Exercise generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export-pdf', methods=['POST'])  # Added missing endpoint
def export_pdf():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content required'}), 400
            
        content = data['content']
        theme = data.get('theme', 'default')
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Customize styles based on theme
        if theme == 'dark':
            styles['Normal'].textColor = colors.white
            doc.background = colors.black
        
        story = []
        for line in content.split('\n'):
            if line.startswith('==='):
                story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.grey))
            elif line.strip():
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 12))
            
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=wellness-report.pdf'
        return response
        
    except Exception as e:
        app.logger.error(f"PDF export error: {str(e)}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

@app.route('/user-exercises', methods=['GET'])
def get_user_exercises():
    try:
        user_id = request.args.get('user_id', 'default')
        personalized_exercises = db.get_user_personalized_exercises(user_id)
        
        return jsonify({
            'success': True,
            'exercises': personalized_exercises,
            'count': len(personalized_exercises)
        })
        
    except Exception as e:
        app.logger.error(f"Error retrieving user exercises: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_conversation():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        
        if user_id not in conversations or not conversations[user_id]:
            return jsonify({'error': 'No conversation found'}), 404
            
        conversation = conversations[user_id]
        
        stats = {
            'message_count': len(conversation),
            'user_messages': sum(1 for msg in conversation if msg['role'] == 'user'),
            'assistant_messages': sum(1 for msg in conversation if msg['role'] == 'assistant'),
            'high_stress_count': sum(
                1 for msg in conversation 
                if msg.get('stress_analysis', {}).get('level', 0) == 1
            ),
            'conversation_duration': (
                datetime.fromisoformat(conversation[-1]['timestamp']) -
                datetime.fromisoformat(conversation[0]['timestamp'])
            ).total_seconds() / 60 if conversation else 0
        }
        
        summary = generate_summary(conversation)
        
        return jsonify({
            'summary': summary,
            'stats': stats,
            'success': True
        })
        
    except Exception as e:
        app.logger.error(f"Summary endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        
        if user_id in conversations:
            del conversations[user_id]
            
        return jsonify({
            'message': 'Conversation cleared',
            'success': True
        })
        
    except Exception as e:
        app.logger.error(f"Clear conversation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        stress_predictor.predict("Test message")
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    setup_logging()
    app.run(debug=True, host='0.0.0.0', port=5000)