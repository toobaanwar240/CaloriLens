# segmentation/yolo_seg.py

import cv2
from ultralytics import YOLO
import numpy as np

class FoodSegmentationYOLO:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def segment(self, image):
        """
        Returns:
            bboxes: list of [x1, y1, x2, y2, conf]
            masks: list of binary masks (H x W)
        """
        results = self.model(image)[0]

        bboxes = []
        masks = []

        if results.masks is None:
            return [], []

        for box, mask in zip(results.boxes, results.masks.data):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy())
            bboxes.append([int(x1), int(y1), int(x2), int(y2), conf])

            mask_np = mask.cpu().numpy().astype("uint8")
            masks.append(mask_np)

        return bboxes, masks
