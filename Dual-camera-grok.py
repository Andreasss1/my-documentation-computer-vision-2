import cv2
from ultralytics import YOLO

# Load the YOLO model (assuming it's a YOLOv8 model saved as 'oppo-ai-model.pt')
model = YOLO('oppo-ai-model.pt')

# Open the two cameras (assuming camera indices 0 and 1; adjust if needed)
cap_upper = cv2.VideoCapture(0)  # Camera for upper screen
cap_lower = cv2.VideoCapture(1)  # Camera for lower screen

# Check if cameras opened successfully
if not cap_upper.isOpened() or not cap_lower.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Set frame size if needed (optional, for better performance)
cap_upper.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_upper.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_lower.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_lower.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit.")

while True:
    # Read frames from both cameras
    ret_upper, frame_upper = cap_upper.read()
    ret_lower, frame_lower = cap_lower.read()

    if not ret_upper or not ret_lower:
        print("Error: Failed to capture frame from one or both cameras.")
        break

    # Run YOLO detection on upper frame
    results_upper = model(frame_upper)
    annotated_upper = results_upper[0].plot()  # Annotate detections

    # Run YOLO detection on lower frame (using the same model)
    results_lower = model(frame_lower)
    annotated_lower = results_lower[0].plot()  # Annotate detections

    # Display the annotated frames in separate windows
    cv2.imshow('Upper Screen (Bubble Detection)', annotated_upper)
    cv2.imshow('Lower Screen (Bubble Detection)', annotated_lower)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
cap_upper.release()
cap_lower.release()
cv2.destroyAllWindows()
