"""
Segmentation Model Wrapper for YOLOv8
Provides consistent interface for object segmentation
"""

import numpy as np
import cv2
from typing import Tuple, List, Union
from ultralytics import YOLO


class SegmentationModel:
    """
    Unified segmentation model interface supporting YOLOv8
    """
    
    def __init__(self, model_type: str = "yolo", model_path: str = "yolov8m-seg.pt"):
        """
        Initialize segmentation model
        
        Args:
            model_type: Type of model ("yolo")
            model_path: Path to model weights or model name
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "yolo":
            self._init_yolo(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_yolo(self, model_path: str):
        """Initialize YOLOv8 segmentation model"""
        try:
            print(f"🔄 Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
    
    def segment(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List, List]:
        """
        Perform segmentation on image
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections
        
        Returns:
            bboxes: List of bounding boxes [x1, y1, x2, y2, confidence, class_id]
            masks: List of binary segmentation masks (numpy arrays)
        """
        if self.model_type == "yolo":
            return self._segment_yolo(image, conf_threshold)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _segment_yolo(self, image: np.ndarray, conf_threshold: float) -> Tuple[List, List]:
        """
        Perform YOLO segmentation
        
        Returns:
            bboxes: List of [x1, y1, x2, y2, confidence, class_id]
            masks: List of binary masks (H, W) numpy arrays
        """
        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        bboxes = []
        masks = []
        
        # Process results
        if len(results) > 0:
            result = results[0]  # Get first result
            
            # Check if segmentation masks exist
            if result.masks is not None:
                # Get boxes
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                # Get masks
                seg_masks = result.masks.data.cpu().numpy()  # (N, H, W)
                
                # Resize masks to original image size
                img_h, img_w = image.shape[:2]
                
                for i in range(len(boxes)):
                    # Bounding box with confidence and class
                    bbox = [
                        float(boxes[i][0]),
                        float(boxes[i][1]),
                        float(boxes[i][2]),
                        float(boxes[i][3]),
                        float(confidences[i]),
                        int(class_ids[i])
                    ]
                    bboxes.append(bbox)
                    
                    # Resize mask to original image size
                    mask = seg_masks[i]
                    mask_resized = cv2.resize(
                        mask,
                        (img_w, img_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Convert to binary mask
                    binary_mask = (mask_resized > 0.5).astype(bool)
                    masks.append(binary_mask)
            
            # If no masks but boxes exist (detection mode)
            elif result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    bbox = [
                        float(boxes[i][0]),
                        float(boxes[i][1]),
                        float(boxes[i][2]),
                        float(boxes[i][3]),
                        float(confidences[i]),
                        int(class_ids[i])
                    ]
                    bboxes.append(bbox)
                    
                    # Create rectangular mask from bbox
                    mask = np.zeros(image.shape[:2], dtype=bool)
                    x1, y1, x2, y2 = map(int, boxes[i])
                    mask[y1:y2, x1:x2] = True
                    masks.append(mask)
        
        return bboxes, masks
    
    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect"""
        if self.model_type == "yolo":
            return list(self.model.names.values())
        return []
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID"""
        if self.model_type == "yolo":
            return self.model.names.get(class_id, f"class_{class_id}")
        return f"class_{class_id}"


# Utility functions for visualization

def visualize_segmentation(image: np.ndarray, bboxes: List, masks: List, 
                          class_names: List[str] = None) -> np.ndarray:
    """
    Visualize segmentation results on image
    
    Args:
        image: Original image (BGR)
        bboxes: List of bounding boxes
        masks: List of segmentation masks
        class_names: Optional list of class names
    
    Returns:
        Annotated image
    """
    overlay = image.copy()
    
    # Color palette
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 128, 0), (0, 128, 255)
    ]
    
    for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
        color = colors[i % len(colors)]
        
        # Draw mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        if len(bbox) >= 5:
            conf = bbox[4]
            class_id = int(bbox[5]) if len(bbox) >= 6 else 0
            
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {conf:.2f}"
            else:
                label = f"Object {i+1}: {conf:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return overlay


def extract_largest_object(image: np.ndarray, bboxes: List, masks: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the largest segmented object from image
    
    Args:
        image: Original image
        bboxes: List of bounding boxes
        masks: List of masks
    
    Returns:
        cropped_image: Image containing only the largest object
        mask: Binary mask of the largest object
    """
    if len(masks) == 0:
        return image, np.ones(image.shape[:2], dtype=bool)
    
    # Find largest mask by area
    areas = [np.sum(mask) for mask in masks]
    largest_idx = np.argmax(areas)
    
    largest_mask = masks[largest_idx]
    largest_bbox = bboxes[largest_idx]
    
    # Crop image to bounding box
    x1, y1, x2, y2 = map(int, largest_bbox[:4])
    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = largest_mask[y1:y2, x1:x2]
    
    # Apply mask (set background to white or black)
    cropped_image[~cropped_mask] = 255  # White background
    
    return cropped_image, cropped_mask