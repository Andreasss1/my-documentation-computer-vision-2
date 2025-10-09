import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO models
model_roi = YOLO('ROI.pt')  # Model for detecting HP screen as ROI
model_bubble = YOLO('bubble.pt')  # Model for detecting bubbles

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to grayscale
    enhanced_gray = clahe.apply(gray)
    
    # Convert back to BGR (for consistency with original frame)
    enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # Detect ROI (HP screen) using the ROI model on enhanced frame
    results_roi = model_roi(enhanced_frame)

    # Assume the ROI is the first detected object (class 0, adjust if needed)
    for result in results_roi:
        boxes = result.boxes
        if len(boxes) > 0:
            # Get the bounding box of the ROI (x1, y1, x2, y2)
            box = boxes[0].xyxy[0].cpu().numpy()  # Take the first detection
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the enhanced frame to the ROI area
            roi_crop = enhanced_frame[y1:y2, x1:x2]
            
            # If ROI is too small, skip
            if roi_crop.shape[0] == 0 or roi_crop.shape[1] == 0:
                continue
            
            # Detect bubbles in the ROI using the bubble model
            results_bubble = model_bubble(roi_crop)
            
            # Draw bubble detections on the cropped ROI
            for bubble_result in results_bubble:
                bubble_boxes = bubble_result.boxes
                for bb in bubble_boxes:
                    bx1, by1, bx2, by2 = map(int, bb.xyxy[0].cpu().numpy())
                    # Draw rectangle on cropped ROI (green for bubbles)
                    cv2.rectangle(roi_crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            # Paste the modified ROI back into the enhanced frame
            enhanced_frame[y1:y2, x1:x2] = roi_crop
            
            # Draw the ROI rectangle on the frame (red for ROI)
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Bubble Detection on HP Screen', enhanced_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
