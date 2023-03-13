import cv2
from models import Yolov4

# Load YOLOv4 model
model = Yolov4(weight_path='yolov4.weights', 
               class_name_path='class_names/coco_classes.txt')

# Open video file
video_path = 'img/cat.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frames
while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    output_img = model.predict(frame)

    # Display output on screen
    cv2.imshow('Output Video', output_img)

    # Write output to video file
    out.write(output_img)

    # Check if user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
