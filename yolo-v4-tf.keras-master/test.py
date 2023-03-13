import cv2
from models import Yolov4
from PIL import Image
import os

# Create the output directory if it doesn't exist
output_dir = 'prediction_result_frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = Yolov4(weight_path='yolov4.weights', 
               class_name_path='class_names/coco_classes.txt')

# Open the video file
cap = cv2.VideoCapture('img/cat.mp4')

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Loop through all frames of the video
frame_num = 0
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Make predictions using the Yolov4 model
    output_img = model.predict(frame)
    
    # Save the output image to a file
    output_path = os.path.join(output_dir, f'frame_{frame_num:06d}.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    frame_num += 1
    
    # Show the output image
    cv2.imshow('Output Image', output_img)
    
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
