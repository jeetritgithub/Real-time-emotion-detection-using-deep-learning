"""
🎭 Advanced Emotion Detection System
Using FER (Facial Expression Recognition) for better accuracy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import logging
import os
from datetime import datetime
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Configuration
EMOTION_MODEL_PATH = 'FER'
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': '#FF6B6B',
    'disgust': '#4ECDC4',
    'fear': '#45B7D1',
    'happy': '#FFA07A',
    'neutral': '#98D8C8',
    'sad': '#F7DC6F',
    'surprise': '#BB8FCE'
}

# Global variables
model = None
face_cascade = None
emotion_stats = {emotion: 0 for emotion in VALID_EMOTIONS}
total_detections = 0
detection_history = []

def load_emotion_model():
    """Load the emotion detection model"""
    global model
    try:
        if os.path.exists(EMOTION_MODEL_PATH):
            model = load_model(EMOTION_MODEL_PATH)
            logger.info("✅ Custom emotion model loaded successfully")
        else:
            logger.warning(f"⚠️  Model file {EMOTION_MODEL_PATH} not found - using mock detection")
    except Exception as e:
        logger.error(f"❌ Failed to load emotion model: {e}")
        model = None

def load_face_cascade():
    """Load OpenCV face cascade"""
    global face_cascade
    try:
        # Try different cascade files
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]

        for cascade_path in cascade_paths:
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ Face cascade loaded successfully")
                return

        logger.warning("⚠️  Face cascade not found - face detection disabled")
    except Exception as e:
        logger.error(f"❌ Failed to load face cascade: {e}")

def preprocess_face(face_img):
    """Preprocess face image for emotion detection"""
    try:
        # Convert to grayscale if needed
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to 48x48 (standard FER2013 size)
        face_img = cv2.resize(face_img, (48, 48))

        # Normalize
        face_img = face_img.astype('float32') / 255.0

        # Add batch and channel dimensions
        face_img = np.expand_dims(face_img, [0, -1])

        return face_img
    except Exception as e:
        logger.error(f"Face preprocessing error: {e}")
        return None

def detect_emotions_fer(frame):
    """Advanced emotion detection using FER model"""
    if model is None:
        # Fallback to mock detection
        return get_mock_emotion_data()

    if face_cascade is None:
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'all_emotions': {e: 0.0 for e in VALID_EMOTIONS},
            'face_count': 0,
            'message': 'Face detection not available'
        }

    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {e: 0.0 for e in VALID_EMOTIONS},
                'face_count': 0,
                'message': 'No faces detected'
            }

        # Process primary face (largest)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # Extract face region
        face_img = gray[y:y+h, x:x+h]

        # Preprocess for model
        processed_face = preprocess_face(face_img)

        if processed_face is None:
            return get_mock_emotion_data()

        # Predict emotions
        predictions = model.predict(processed_face, verbose=0)[0]

        # Create emotion dictionary
        emotions_dict = {emotion: float(pred) for emotion, pred in zip(VALID_EMOTIONS, predictions)}

        # Get dominant emotion
        dominant_emotion = max(emotions_dict, key=emotions_dict.get)
        confidence = emotions_dict[dominant_emotion]

        # Prepare response
        response = {
            'emotion': dominant_emotion,
            'confidence': round(float(confidence), 4),
            'all_emotions': {k: round(float(v), 4) for k, v in emotions_dict.items()},
            'face_count': len(faces),
            'faces': [{
                'id': 1,
                'emotion': dominant_emotion,
                'confidence': round(float(confidence), 4),
                'all_emotions': {k: round(float(v), 4) for k, v in emotions_dict.items()},
                'facial_area': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'detection_confidence': 0.9
            }],
            'facial_area': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'detection_confidence': 0.9
        }

        logger.info(f"Detected {len(faces)} faces. Primary: {dominant_emotion} ({confidence:.2%})")
        return response

    except Exception as e:
        logger.error(f"FER detection error: {e}")
        logger.error(traceback.format_exc())
        return get_mock_emotion_data()

def get_mock_emotion_data():
    """Return mock emotion data for fallback"""
    import random
    mock_emotion = random.choice(VALID_EMOTIONS)
    mock_confidence = random.uniform(0.3, 0.8)

    mock_emotions = {emotion: 0.0 for emotion in VALID_EMOTIONS}
    mock_emotions[mock_emotion] = mock_confidence

    # Distribute remaining probability
    remaining = 1.0 - mock_confidence
    other_emotions = [e for e in VALID_EMOTIONS if e != mock_emotion]
    for emotion in other_emotions:
        mock_emotions[emotion] = remaining / len(other_emotions)

    logger.warning(f"🔄 Using mock emotion detection: {mock_emotion} ({mock_confidence:.1%})")

    return {
        'emotion': mock_emotion,
        'confidence': round(mock_confidence, 4),
        'all_emotions': {k: round(v, 4) for k, v in mock_emotions.items()},
        'face_count': 1,
        'faces': [{
            'id': 1,
            'emotion': mock_emotion,
            'confidence': round(mock_confidence, 4),
            'all_emotions': {k: round(v, 4) for k, v in mock_emotions.items()},
            'facial_area': {'x': 100, 'y': 100, 'w': 200, 'h': 200},
            'region': {'x': 100, 'y': 100, 'w': 200, 'h': 200},
            'detection_confidence': 0.8
        }],
        'facial_area': {'x': 100, 'y': 100, 'w': 200, 'h': 200},
        'region': {'x': 100, 'y': 100, 'w': 200, 'h': 200},
        'detection_confidence': 0.8,
        'message': 'Using mock emotion detection (model unavailable)'
    }

# Initialize components
load_emotion_model()
load_face_cascade()

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok' if model is not None else 'degraded',
        'message': 'Advanced emotion detection service',
        'model_loaded': model is not None,
        'face_cascade_loaded': face_cascade is not None,
        'supported_emotions': VALID_EMOTIONS,
        'emotion_colors': EMOTION_COLORS
    })

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        frame_file = request.files['frame']

        # Read image
        image_bytes = frame_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Resize for performance
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        # Detect emotions
        response = detect_emotions_fer(frame)

        # Update statistics
        if response.get('face_count', 0) > 0:
            emotion = response['emotion'].lower()
            if emotion in emotion_stats:
                emotion_stats[emotion] += 1

            global total_detections
            total_detections += 1

            # Store in history
            detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'confidence': response['confidence']
            })
            if len(detection_history) > 100:
                detection_history.pop(0)

        return jsonify(response), 200

    except Exception as e:
        logger.error(f'Request error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    return jsonify({
        'emotions': VALID_EMOTIONS,
        'colors': EMOTION_COLORS,
        'count': len(VALID_EMOTIONS)
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    total = sum(emotion_stats.values())
    percentages = {
        emotion: round((count / total) * 100, 2) if total > 0 else 0.0
        for emotion, count in emotion_stats.items()
    }

    return jsonify({
        'total_detections': total_detections,
        'emotion_distribution': emotion_stats,
        'emotion_percentages': percentages,
        'most_common': max(emotion_stats, key=emotion_stats.get) if total > 0 else 'none',
        'detection_history': detection_history[-20:]
    })

if __name__ == '__main__':
    logger.info("🎭 Advanced Emotion Detection Server Starting...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)