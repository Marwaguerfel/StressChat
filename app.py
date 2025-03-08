from flask import Flask, render_template, request, jsonify, make_response, send_file
from groq import Groq
import torch
from model.predictor import StressPredictor
from DataBaseSteup import Database 
from EmbadingRet import EmbeddingRetrieval  
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
from openai import OpenAI
from gtts import gTTS
import soundfile as sf 
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset
from ultralytics import YOLO
import cv2
import numpy as np
import base64







face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotion_model = YOLO('best.pt')  




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
openai_client = OpenAI(api_key=GROQ_API_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stress_predictor = StressPredictor(device)
conversations = {}


def setup_logging():
    """Configure application logging"""
    if not app.debug:
        os.makedirs('logs', exist_ok=True)
        try:
            file_handler = RotatingFileHandler(
                'logs/chatbot.log',
                maxBytes=10240,
                backupCount=10,
                delay=True  # Only open the file when first log is emitted
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
        except PermissionError:
            # Fall back to console logging if file is locked
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s'
            ))
            console_handler.setLevel(logging.INFO)
            app.logger.addHandler(console_handler)
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






def detect_emotion_from_image(image_data):
    """Detect emotion from base64 encoded image data with face detection preprocessing"""
    try:
        # Decode base64 image
        if ',' in image_data:
            base64_data = image_data.split(',')[1]
        else:
            base64_data = image_data
            
        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            app.logger.error("Failed to decode image")
            return None
        
        app.logger.info(f"Image shape: {img.shape}")
        
        # Save original image for debugging
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        cv2.imwrite(debug_path, img)
        
        # Try face detection first
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        target_img = img  # Default to using the entire image
        
        if len(faces) > 0:
            app.logger.info(f"Found {len(faces)} faces in image")
            
            # Process the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add some margin around the face
            margin = int(0.2 * min(w, h))
            y_start = max(0, y - margin)
            y_end = min(img.shape[0], y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(img.shape[1], x + w + margin)
            face_img = img[y_start:y_end, x_start:x_end]
            
            # Save face image for debugging
            face_path = os.path.join(debug_dir, f"face_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            cv2.imwrite(face_path, face_img)
            
            # Use the face image for emotion detection
            target_img = face_img
        else:
            app.logger.warning("No face detected in image")
            # We'll still try emotion detection on the full image as fallback
        
        # Run emotion detection with lower confidence threshold
        results = emotion_model.predict(source=target_img, verbose=True, conf=0.1)
        
        if results and len(results) > 0:
            app.logger.info(f"Detection results: {len(results)}")
            
            # Extract the top emotion detection result
            result = results[0]
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                app.logger.info(f"Found {len(result.boxes)} detections")
                
                # Get class indices and confidence scores
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                
                if len(cls) > 0:
                    app.logger.info(f"Class indices: {cls}, Confidences: {conf}")
                    
                    # Get the class with highest confidence
                    top_class_idx = cls[np.argmax(conf)]
                    emotion_labels = result.names  # Get mapping of class indices to labels
                    
                    app.logger.info(f"Top class: {top_class_idx}, Labels: {emotion_labels}")
                    
                if isinstance(emotion_labels, dict):
                    class_key = int(top_class_idx)  # Convert float to int
                    if class_key in emotion_labels:  # Check directly without converting to string
                        emotion = emotion_labels[class_key]
                        confidence = float(np.max(conf))
                        
                        app.logger.info(f"Detected emotion: {emotion} with confidence {confidence}")
                        
                        return {
                            "emotion": emotion,
                            "confidence": confidence
                        }
                    else:
                        app.logger.warning(f"Class {top_class_idx} not found in labels: {emotion_labels}")
                else:
                    app.logger.warning("No class indices found in detection")
            else:
                app.logger.warning("No boxes in detection result")
        else:
            app.logger.warning("No detection results returned by model")
        
        return None
    except Exception as e:
        app.logger.error(f"Emotion detection error: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return None














def get_groq_response(user_message, stress_level, confidence, user_id='default', original_language=None):
    """Get response from Groq API with appropriate context"""
    try:
        history = conversations.get(user_id, [])
        
        # Get a friendly, human-like system prompt based on stress level
        messages = [{"role": "system", "content": get_stress_prompt(stress_level) + 
                    " Respond in a warm, conversational tone like a supportive friend. Use casual language and occasional contractions. Keep responses brief and friendly."}]
        
        # Add system message for multilingual support
        if original_language and original_language != 'en':
            messages.append({
                "role": "system", 
                "content": f"The user is communicating in {original_language}. Their messages are being translated to English for you. Your responses will be translated back to {original_language} for them."
            })
        
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
            model="qwen-2.5-32b",
            temperature=0.8,  # Slightly increased for more natural variation
            max_tokens=250,   # Reduced for shorter responses
            top_p=0.9
        )
        
        response = completion.choices[0].message.content
        
        if user_id not in conversations:
            conversations[user_id] = []
            
        # Create the user message entry with language information if provided
        user_msg = {
            "timestamp": datetime.now().isoformat(),
            "message": user_message,
            "role": "user",
            "stress_analysis": {"level": stress_level, "confidence": confidence}
        }
        
        if original_language:
            user_msg["original_language"] = original_language
            
        conversations[user_id].extend([
            user_msg,
            {
                "timestamp": datetime.now().isoformat(),
                "message": response,
                "role": "assistant"
            }
        ])
        
        return response
        
    except Exception as e:
        app.logger.error(f"Groq API error: {str(e)}")
        return "I'm having trouble connecting right now. Can you try again in a moment?"

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
        original_language = data.get('original_language', 'en')

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        stress_result = stress_predictor.predict(user_message)
        response = get_groq_response(
            user_message,
            stress_result['stress_level'],
            stress_result['confidence'],
            user_id,
            original_language
        )

        return jsonify({
            'message': user_message,
            'response': response,
            'stress_analysis': stress_result,
            'original_language': original_language
        })

    except Exception as e:
        app.logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    

















@app.route('/test-emotion', methods=['GET'])
def test_emotion():
    """Test emotion detection using direct camera access"""
    try:
        # Use the same YOLO model with direct camera access
        results = emotion_model.predict(source=0, save=True, save_dir="output")
        
        # Return success message
        return jsonify({
            'success': True,
            'message': 'Emotion detection test completed, check output directory for results'
        })
    except Exception as e:
        app.logger.error(f"Test emotion error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    








@app.route('/generate-exercise', methods=['POST'])
async def generate_exercise_from_summary():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'error': 'User ID required'}), 400
            
        user_id = data['user_id']
        target_language = data.get('language', 'en')
        
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
        
        # Note: For simplicity, we're not translating the exercise on the backend
        # The frontend will handle translation of the exercise content
        
        app.logger.info(f"Generated RAG exercise: {new_exercise.get('title', 'Unknown')}")
        
        return jsonify({
            'success': True,
            'exercise': new_exercise,
            'method': 'rag',
            'target_language': target_language
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
    




# New route for standalone emotion detection
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        app.logger.info(f"Received detect-emotion request: {data is not None}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        if 'image_data' not in data:
            return jsonify({'error': 'No image_data field in request'}), 400
            
        image_data = data['image_data']
        
        # Log the start of the image data to help debug
        app.logger.info(f"Image data type: {type(image_data)}, starts with: {image_data[:50] if isinstance(image_data, str) else 'not a string'}")
        
        # Validate the image data
        if not isinstance(image_data, str):
            return jsonify({'error': f'Invalid image data type: {type(image_data)}'}), 400
            
        if not image_data.startswith('data:image'):
            return jsonify({'error': 'Invalid image data format, must start with data:image'}), 400
            
        # Check for reasonable length
        if len(image_data) < 1000:
            return jsonify({'error': 'Image data too short'}), 400
            
        emotion_result = detect_emotion_from_image(image_data)
        
        if not emotion_result:
            return jsonify({'success': False, 'error': 'No emotions detected'}), 200
            
        return jsonify({
            'success': True,
            'emotion_analysis': emotion_result
        })
        
    except Exception as e:
        app.logger.error(f"Emotion detection endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

# New route for video emotion detection
@app.route('/video-emotion', methods=['POST'])
def video_emotion():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video']
        user_id = request.form.get('user_id', 'default')
        
        # Save the video temporarily
        temp_path = f"temp_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
        video_file.save(temp_path)
        
        # Process video for emotion detection
        cap = cv2.VideoCapture(temp_path)
        
        emotions = []
        frame_count = 0
        sample_rate = 10  # Process every 10th frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
                
            # Run emotion detection on frame
            results = emotion_model.predict(source=frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Get top detection
                    cls = result.boxes.cls.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    
                    if len(cls) > 0:
                        top_idx = np.argmax(conf)
                        top_class_idx = cls[top_idx]
                        emotion_labels = result.names
                        
                        if isinstance(emotion_labels, dict) and str(int(top_class_idx)) in emotion_labels:
                            emotions.append({
                                "emotion": emotion_labels[str(int(top_class_idx))],
                                "confidence": float(conf[top_idx]),
                                "frame": frame_count
                            })
        
        cap.release()
        os.remove(temp_path)
        
        if not emotions:
            return jsonify({'error': 'No emotions detected in video'}), 400
            
        # Aggregate emotions
        emotion_counts = {}
        for e in emotions:
            emotion = e["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
            
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        # Calculate average confidence for dominant emotion
        avg_confidence = sum(e["confidence"] for e in emotions if e["emotion"] == dominant_emotion[0]) / dominant_emotion[1]
        
        emotion_result = {
            "emotion": dominant_emotion[0],
            "confidence": avg_confidence,
            "frames_analyzed": len(emotions),
            "emotion_distribution": emotion_counts
        }
        
        return jsonify({
            'success': True,
            'emotion_analysis': emotion_result
        })
        
    except Exception as e:
        app.logger.error(f"Video emotion detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500


















@app.route('/generate-tts', methods=['POST'])
async def generate_tts():
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'en')  # Default to English

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Log for debugging
        app.logger.info(f"Generating TTS for text: '{text}' in language: '{language}'")

        # Use gTTS with language-specific voice
        if language not in ['en', 'fr', 'es']:
            app.logger.warning(f"Language '{language}' not supported; defaulting to English")
            language = 'en'

        # Generate TTS with gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)

        # Return MP3 audio
        return send_file(
            audio_io,
            mimetype='audio/mp3',
            as_attachment=False,
            download_name='tts_response.mp3'
        )

    except Exception as e:
        app.logger.error(f"TTS generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500



# New endpoint for voice input
@app.route('/voice-chat', methods=['POST'])
async def voice_chat():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'default')
        original_language = request.form.get('original_language', 'en')

        # Save the audio temporarily
        temp_path = f"temp_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
        audio_file.save(temp_path)

        # Transcribe audio using Whisper
        with open(temp_path, 'rb') as f:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                language=original_language  # Use detected or preferred language
            )

        # Clean up temporary file
        os.remove(temp_path)

        transcribed_text = transcription.text.strip()
        if not transcribed_text:
            return jsonify({'error': 'No speech detected'}), 400

        # Process transcribed text as a regular chat message
        stress_result = stress_predictor.predict(transcribed_text)
        response = get_groq_response(
            transcribed_text,
            stress_result['stress_level'],
            stress_result['confidence'],
            user_id,
            original_language
        )

        return jsonify({
            'message': transcribed_text,
            'response': response,
            'stress_analysis': stress_result,
            'original_language': original_language
        })

    except Exception as e:
        app.logger.error(f"Voice chat error: {str(e)}")
        return jsonify({'error': f'Failed to process voice input: {str(e)}'}), 500





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
        target_language = data.get('language', 'en')
        
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
            ).total_seconds() / 60 if conversation else 0,
            'language_distribution': {}
        }
        
        # Add language statistics
        for msg in conversation:
            if msg['role'] == 'user' and 'original_language' in msg:
                lang = msg['original_language']
                if lang not in stats['language_distribution']:
                    stats['language_distribution'][lang] = 0
                stats['language_distribution'][lang] += 1
        
        summary = generate_summary(conversation)
        
        # Note: For simplicity, we're not translating the summary on the backend
        # The frontend will handle translation of the summary content
        
        return jsonify({
            'summary': summary,
            'stats': stats,
            'success': True,
            'target_language': target_language
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

@app.route('/interface-translations', methods=['GET'])
def get_interface_translations():
    """Endpoint to provide translations for the UI"""
    try:
        language = request.args.get('language', 'en')
        
        # Basic translations for common UI elements
        translations = {
            'en': {

                'title': 'Mental Wellness Assistant',
                'subtitle': 'Here to support you 24/7',
                'stress_analysis': 'Stress Analysis',
                'current_stress': 'Current stress level:',
                'not_analyzed': 'Not analyzed',
                'toggle_theme': 'Toggle Theme',
                'language_settings': 'Language Settings',
                'summary': 'Summary',
                'generate_exercise': 'Generate Exercise',
                'export_pdf': 'Export PDF',
                'send': 'Send',
                'type_message': 'Type your message here...',
                'generating': 'Generating a personalized exercise...',
                'pdf_preparing': 'Preparing your PDF export...',
                'first_message':'Hello! I\'m your wellness assistant. How are you feeling today?'
            },
            'fr': {
                'title': 'Assistant de Bien-être Mental',
                'subtitle': 'Là pour vous soutenir 24/7',
                'stress_analysis': 'Analyse de Stress',
                'current_stress': 'Niveau de stress actuel:',
                'not_analyzed': 'Non analysé',
                'toggle_theme': 'Changer de Thème',
                'language_settings': 'Paramètres de Langue',
                'summary': 'Résumé',
                'generate_exercise': 'Générer un Exercice',
                'export_pdf': 'Exporter en PDF',
                'send': 'Envoyer',
                'type_message': 'Tapez votre message ici...',
                'generating': 'Génération d\'un exercice personnalisé...',
                'pdf_preparing': 'Préparation de votre export PDF...',
                'first_message':'Bonjour ! Je suis votre assistante bien-être. Comment vous sentez-vous aujourd\'hui ?'

            },
            'es': {
                'title': 'Asistente de Bienestar Mental',
                'subtitle': 'Aquí para apoyarte 24/7',
                'stress_analysis': 'Análisis de Estrés',
                'current_stress': 'Nivel de estrés actual:',
                'not_analyzed': 'No analizado',
                'toggle_theme': 'Cambiar Tema',
                'language_settings': 'Configuración de Idioma',
                'summary': 'Resumen',
                'generate_exercise': 'Generar Ejercicio',
                'export_pdf': 'Exportar PDF',
                'send': 'Enviar',
                'type_message': 'Escribe tu mensaje aquí...',
                'generating': 'Generando un ejercicio personalizado...',
                'pdf_preparing': 'Preparando tu exportación PDF...',
                'first_mesaage':'Hola, soy tu asistente de bienestar. ¿Cómo te sientes hoy?'
            },
            # Add more languages as needed
        }
        
        if language in translations:
            return jsonify({
                'success': True,
                'translations': translations[language]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Language not supported',
                'fallback': translations['en']  # Return English as fallback
            })
            
    except Exception as e:
        app.logger.error(f"Translation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_logging()
    app.run(debug=True, host='0.0.0.0', port=5000)