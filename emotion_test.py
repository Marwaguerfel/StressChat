from ultralytics import YOLO
from PIL import Image 
import cv2

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