import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from datetime import datetime

class DualCameraYOLODetector:
    def __init__(self, model_path="oppo-ai-model.pt", camera1_id=0, camera2_id=1):
        """
        Initialize dual camera YOLO detector
        Args:
            model_path: Path to YOLO model file
            camera1_id: Camera ID for upper screen detection
            camera2_id: Camera ID for lower screen detection
        """
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize cameras
        self.cap1 = cv2.VideoCapture(camera1_id)  # Upper screen camera
        self.cap2 = cv2.VideoCapture(camera2_id)  # Lower screen camera
        
        # Set camera properties for better quality
        for cap in [self.cap1, self.cap2]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if cameras are opened
        if not self.cap1.isOpened():
            raise Exception(f"Cannot open camera {camera1_id}")
        if not self.cap2.isOpened():
            raise Exception(f"Cannot open camera {camera2_id}")
        
        print("Cameras initialized successfully!")
        
        # Detection results storage
        self.results1 = None
        self.results2 = None
        self.frame1 = None
        self.frame2 = None
        
        # Threading locks
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()
        
        # Detection statistics
        self.defect_count_cam1 = 0
        self.defect_count_cam2 = 0
        self.running = True

    def detect_camera1(self):
        """Detection thread for camera 1 (upper screen)"""
        while self.running:
            ret, frame = self.cap1.read()
            if not ret:
                continue
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5)  # Confidence threshold 0.5
            
            # Draw detection results
            annotated_frame = results[0].plot()
            
            # Count defects
            defect_count = len(results[0].boxes) if results[0].boxes is not None else 0
            
            with self.lock1:
                self.frame1 = annotated_frame
                self.results1 = results[0]
                self.defect_count_cam1 = defect_count
            
            time.sleep(0.03)  # Small delay to prevent overloading

    def detect_camera2(self):
        """Detection thread for camera 2 (lower screen)"""
        while self.running:
            ret, frame = self.cap2.read()
            if not ret:
                continue
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5)  # Confidence threshold 0.5
            
            # Draw detection results
            annotated_frame = results[0].plot()
            
            # Count defects
            defect_count = len(results[0].boxes) if results[0].boxes is not None else 0
            
            with self.lock2:
                self.frame2 = annotated_frame
                self.results2 = results[0]
                self.defect_count_cam2 = defect_count
            
            time.sleep(0.03)  # Small delay to prevent overloading

    def add_info_overlay(self, frame, camera_name, defect_count):
        """Add information overlay on frame"""
        # Add background for text
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        
        # Add text information
        cv2.putText(frame, f"Camera: {camera_name}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Defects: {defect_count}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if defect_count == 0 else (0, 0, 255), 2)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (15, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        """Main execution loop"""
        print("Starting dual camera detection...")
        print("Press 'q' to quit, 's' to save current frames")
        
        # Start detection threads
        thread1 = threading.Thread(target=self.detect_camera1)
        thread2 = threading.Thread(target=self.detect_camera2)
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        
        try:
            while True:
                display_frame1 = None
                display_frame2 = None
                
                # Get frames from both cameras
                with self.lock1:
                    if self.frame1 is not None:
                        display_frame1 = self.frame1.copy()
                        defects1 = self.defect_count_cam1
                
                with self.lock2:
                    if self.frame2 is not None:
                        display_frame2 = self.frame2.copy()
                        defects2 = self.defect_count_cam2
                
                # Display frames if available
                if display_frame1 is not None:
                    # Add overlay information
                    display_frame1 = self.add_info_overlay(display_frame1, "Upper Screen", defects1)
                    cv2.imshow('Camera 1 - Upper Screen', display_frame1)
                
                if display_frame2 is not None:
                    # Add overlay information
                    display_frame2 = self.add_info_overlay(display_frame2, "Lower Screen", defects2)
                    cv2.imshow('Camera 2 - Lower Screen', display_frame2)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frames
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if display_frame1 is not None:
                        cv2.imwrite(f"upper_screen_{timestamp}.jpg", display_frame1)
                    if display_frame2 is not None:
                        cv2.imwrite(f"lower_screen_{timestamp}.jpg", display_frame2)
                    print(f"Frames saved with timestamp: {timestamp}")
                
                # Print detection summary every 5 seconds
                if int(time.time()) % 5 == 0:
                    total_defects = defects1 + defects2
                    print(f"Detection Summary - Upper: {defects1}, Lower: {defects2}, Total: {total_defects}")
                    time.sleep(1)  # Prevent multiple prints in same second
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        time.sleep(0.1)  # Give threads time to stop
        
        # Release cameras
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        print("Cleanup completed!")

def main():
    """Main function"""
    try:
        # Initialize detector
        detector = DualCameraYOLODetector(
            model_path="oppo-ai-model.pt",
            camera1_id=0,  # Change this if your upper camera has different ID
            camera2_id=1   # Change this if your lower camera has different ID
        )
        
        # Start detection
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. oppo-ai-model.pt file exists")
        print("2. Both cameras are connected and accessible")
        print("3. Required libraries are installed: pip install ultralytics opencv-python")

if __name__ == "__main__":
    main()
