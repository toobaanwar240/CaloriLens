import torch
import numpy as np
import cv2


class FoodSegmentationSAM:
    def __init__(self, checkpoint_path="mobile_sam.pt", model_type="vit_t"):
        """
        Initialize MobileSAM for food segmentation
        
        Installation:
        pip install git+https://github.com/ChaoningZhang/MobileSAM.git
        
        Download weights:
        wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Try MobileSAM first
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(self.device)
            self.predictor = SamPredictor(self.sam)
            
            print(f"✓ MobileSAM loaded on {self.device}")
            
        except ImportError:
            # Fallback to regular SAM
            print("MobileSAM not found, using regular SAM...")
            from segment_anything import sam_model_registry, SamPredictor
            
            # For regular SAM, use these model types:
            model_type = "vit_b"  # or "vit_l", "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(self.device)
            self.predictor = SamPredictor(self.sam)
            
            print(f"✓ Regular SAM loaded on {self.device}")

    def segment(self, image):
        """
        Automatic segmentation using SAM
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            bboxes: list of [x1, y1, x2, y2, confidence]
            mask_list: list of binary masks (uint8)
        """
        
        # Import here to handle both MobileSAM and regular SAM
        try:
            from mobile_sam import SamAutomaticMaskGenerator
        except ImportError:
            from segment_anything import SamAutomaticMaskGenerator
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=16,          # Fast: fewer points
            pred_iou_thresh=0.90,        # High quality threshold
            stability_score_thresh=0.92,  # Stability threshold
            min_mask_region_area=3000    # Remove small noise (pixels)
        )
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        if len(masks) == 0:
            print("⚠ No masks found!")
            return [], []
        
        # Extract bboxes and masks
        bboxes = []
        mask_list = []
        
        for m in masks:
            # Get bbox [x, y, w, h]
            x, y, w, h = m["bbox"]
            
            # Convert to [x1, y1, x2, y2, confidence]
            bbox = [int(x), int(y), int(x + w), int(y + h), float(m.get("predicted_iou", 1.0))]
            bboxes.append(bbox)
            
            # Get binary mask
            mask_list.append(m["segmentation"].astype(np.uint8))
        
        print(f"✓ Found {len(masks)} segments")
        
        return bboxes, mask_list
