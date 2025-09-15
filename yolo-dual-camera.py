import cv2
import torch
from ultralytics import YOLO
import threading
import numpy as np
from datetime import datetime

class DualCameraYOLO:
    def __init__(self, model_path="best.pt", camera1_id=0, camera2_id=1):
        """
        Initialize dual camera YOLO detection
        
        Args:
            model_path (str): Path to YOLO model file
            camera1_id (int): First camera source ID
            camera2_id (int): Second camera source ID
        """
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Camera settings
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        
        # Initialize cameras
        self.cap1 = None
        self.cap2 = None
        
        # Control variables
        self.running = False
        self.threads = []
        
        # Detection results storage
        self.results1 = None
        self.results2 = None
        
    def initialize_cameras(self):
        """Initialize both cameras"""
        print(f"Initializing camera {self.camera1_id}...")
        self.cap1 = cv2.VideoCapture(self.camera1_id)
        if not self.cap1.isOpened():
            print(f"Error: Cannot open camera {self.camera1_id}")
            return False
            
        print(f"Initializing camera {self.camera2_id}...")
        self.cap2 = cv2.VideoCapture(self.camera2_id)
        if not self.cap2.isOpened():
            print(f"Error: Cannot open camera {self.camera2_id}")
            self.cap1.release()
            return False
        
        # Set camera properties (optional)
        for cap in [self.cap1, self.cap2]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Both cameras initialized successfully!")
        return True
    
    def detect_and_display(self, cap, camera_name, window_pos_x=0):
        """
        Detection and display function for single camera
        
        Args:
            cap: Camera capture object
            camera_name (str): Name for display window
            window_pos_x (int): X position for window
        """
        window_name = f"YOLO Detection - {camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, window_pos_x, 0)
        cv2.resizeWindow(window_name, 640, 480)
        
        fps_counter = 0
        fps_start_time = datetime.now()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Cannot read frame from {camera_name}")
                break
            
            try:
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Draw detection results
                annotated_frame = results[0].plot()
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    fps_end_time = datetime.now()
                    fps = 30 / (fps_end_time - fps_start_time).total_seconds()
                    fps_start_time = fps_end_time
                
                # Add FPS and detection info to frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Count detections
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                cv2.putText(annotated_frame, f"Detections: {detections}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, camera_name, 
                           (10, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow(window_name, annotated_frame)
                
                # Store results for external access
                if camera_name == "Camera 0":
                    self.results1 = results[0]
                else:
                    self.results2 = results[0]
                
            except Exception as e:
                print(f"Error during detection on {camera_name}: {e}")
                # Display original frame if detection fails
                cv2.imshow(window_name, frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyWindow(window_name)
    
    def start_detection(self):
        """Start dual camera detection in separate threads"""
        if not self.initialize_cameras():
            return False
        
        self.running = True
        print("\nStarting dual camera YOLO detection...")
        print("Press 'q' in any window to quit")
        print("=" * 50)
        
        # Create and start threads for each camera
        thread1 = threading.Thread(
            target=self.detect_and_display, 
            args=(self.cap1, "Camera 0", 0)
        )
        thread2 = threading.Thread(
            target=self.detect_and_display, 
            args=(self.cap2, "Camera 1", 700)
        )
        
        self.threads = [thread1, thread2]
        
        thread1.start()
        thread2.start()
        
        return True
    
    def stop_detection(self):
        """Stop detection and cleanup resources"""
        print("\nStopping detection...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        # Release cameras
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        
        cv2.destroyAllWindows()
        print("Detection stopped and resources cleaned up.")
    
    def get_detection_results(self):
        """Get current detection results from both cameras"""
        return {
            'camera_0': self.results1,
            'camera_1': self.results2
        }

def main():
    """Main function to run dual camera YOLO detection"""
    # Initialize dual camera YOLO detector
    detector = DualCameraYOLO(
        model_path="best.pt",  # Path to your YOLO model
        camera1_id=0,          # First camera ID
        camera2_id=1           # Second camera ID
    )
    
    try:
        # Start detection
        if detector.start_detection():
            # Keep main thread alive until user stops
            while detector.running:
                # Optional: Print detection statistics
                import time
                time.sleep(5)  # Print every 5 seconds
                
                results = detector.get_detection_results()
                if results['camera_0'] and results['camera_1']:
                    det1 = len(results['camera_0'].boxes) if results['camera_0'].boxes else 0
                    det2 = len(results['camera_1'].boxes) if results['camera_1'].boxes else 0
                    print(f"Camera 0: {det1} detections | Camera 1: {det2} detections")
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    main()
