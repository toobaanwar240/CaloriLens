from ultralytics import YOLO
import cv2
import numpy as np

# Use medium for best accuracy/speed balance
model = YOLO('yolov8m-seg.pt')  # 25MB - better accuracy

# Segment food with high confidence
results = model('food_plate.jpg', conf=0.3, iou=0.5)

# Extract precise masks for portion calculation
for result in results:
    masks = result.masks.data  # pixel-perfect masks
    boxes = result.boxes.xyxy  # bounding boxes
    areas = result.masks.data.sum(dim=(1,2))  # pixel areas