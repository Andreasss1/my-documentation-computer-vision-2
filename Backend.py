"""
YOLO Object Detection Backend Server
Jalankan: python app.py

Requirements:
pip install flask flask-socketio flask-cors ultralytics opencv-python
"""

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO model
MODEL_PATH = 'ai-model.pt'
model = YOLO(MODEL_PATH)

# Global variables
camera = None
is_running = False

def init_camera():
    """Initialize camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 0 untuk webcam default
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def detect_objects_from_frame(frame):
    """
    Detect objects using YOLO model
    Returns: list of detections
    """
    results = model(frame, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Hanya ambil deteksi 'Gelembung Standar' atau 'Gelembung Solid'
            if class_name in ['Gelembung Standar', 'Gelembung Solid']:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection = {
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                }
                detections.append(detection)
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label background
        label = f"{det['class']} {det['confidence']*100:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x, y - text_height - 10), 
                     (x + text_width, y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def generate_frames():
    """Generator function for video streaming"""
    global is_running
    camera = init_camera()
    
    while is_running:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect objects
        detections = detect_objects_from_frame(frame)
        
        # Draw detections on frame
        frame_with_detections = draw_detections(frame.copy(), detections)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_detections)
        frame_bytes = buffer.tobytes()
        
        # Emit detections via WebSocket
        socketio.emit('detections', json.dumps(detections))
        
        # Yield frame for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """Serve a simple status page"""
    return """
    <html>
        <head><title>YOLO Detection Server</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>YOLO Object Detection Server</h1>
            <p>Status: <span style="color: green;">Running</span></p>
            <p>Model: ai-model.pt</p>
            <p>WebSocket endpoint: ws://localhost:5000</p>
            <h2>Endpoints:</h2>
            <ul>
                <li><a href="/video_feed">/video_feed</a> - Video stream with detections</li>
            </ul>
            <h2>Live Video Feed:</h2>
            <img src="/video_feed" style="border: 2px solid #333; max-width: 100%;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global is_running
    is_running = True
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_detection')
def handle_start_detection():
    """Start detection process"""
    global is_running
    is_running = True
    emit('detection_status', {'status': 'started'})

@socketio.on('stop_detection')
def handle_stop_detection():
    """Stop detection process"""
    global is_running
    is_running = False
    emit('detection_status', {'status': 'stopped'})

if __name__ == '__main__':
    print("=" * 50)
    print("YOLO Object Detection Server")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Server: http://localhost:5000")
    print(f"WebSocket: ws://localhost:5000")
    print("=" * 50)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    finally:
        release_camera()
        print("Server stopped. Camera released.")
