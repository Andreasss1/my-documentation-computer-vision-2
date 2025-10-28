from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import time
from collections import deque

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO('new-sg-model.pt')

# Global variables
detection_counter = 0
status = "PASS"
status_color = "green"
ng_start_time = None
NG_DURATION = 5  # durasi NG dalam detik
CONFIDENCE_THRESHOLD = 0.4
CONSECUTIVE_DETECTIONS = 4

# Deque untuk tracking deteksi berturut-turut
recent_detections = deque(maxlen=10)

def generate_frames():
    global detection_counter, status, status_color, ng_start_time, recent_detections
    
    camera = cv2.VideoCapture(0)
    
    # Set resolusi kamera (opsional)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Cek apakah masih dalam periode NG 5 detik
        current_time = time.time()
        if ng_start_time and (current_time - ng_start_time) >= NG_DURATION:
            # Reset setelah 5 detik
            status = "PASS"
            status_color = "green"
            detection_counter = 0
            ng_start_time = None
            recent_detections.clear()
        
        # Jangan update deteksi jika masih dalam periode NG
        if ng_start_time is None:
            # Run YOLO inference
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Cek apakah ada deteksi 'Gelembung Standar'
            gelembung_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Warna box: merah jika Gelembung Standar, hijau untuk lainnya
                    if class_name == 'Gelembung Standar' and confidence > CONFIDENCE_THRESHOLD:
                        color = (0, 0, 255)  # Merah
                        gelembung_detected = True
                    else:
                        color = (0, 255, 0)  # Hijau
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update detection counter
            if gelembung_detected:
                detection_counter += 1
                recent_detections.append(1)
            else:
                detection_counter = 0  # Reset counter
                recent_detections.append(0)
            
            # Cek apakah sudah 4 kali berturut-turut
            if detection_counter >= CONSECUTIVE_DETECTIONS:
                status = "NG"
                status_color = "red"
                ng_start_time = current_time
                detection_counter = 0  # Reset untuk deteksi berikutnya
        else:
            # Masih dalam periode NG, tetap tampilkan frame dengan bounding box
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 0, 255) if class_name == 'Gelembung Standar' else (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global status, status_color, detection_counter, ng_start_time
    
    remaining_time = 0
    if ng_start_time:
        elapsed = time.time() - ng_start_time
        remaining_time = max(0, NG_DURATION - elapsed)
    
    return jsonify({
        'status': status,
        'color': status_color,
        'counter': detection_counter,
        'remaining_time': round(remaining_time, 1)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
