# segmentation/sam_seg.py

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamPredictor


class FoodSegmentationSAM:
    def __init__(self, checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)

        self.predictor = SamPredictor(self.sam)

    def segment(self, image):
        """
        SAM requires image embedding + prompts.
        We use a fully automatic mask generator.
        """

        from segment_anything import SamAutomaticMaskGenerator

        mask_generator = SamAutomaticMaskGenerator(self.sam)

        masks = mask_generator.generate(image)

        if len(masks) == 0:
            return [], []

        bboxes = []
        mask_list = []

        for m in masks:
            x, y, w, h = m["bbox"]
            bbox = [int(x), int(y), int(x + w), int(y + h), 1.0]
            bboxes.append(bbox)

            mask_list.append(m["segmentation"].astype("uint8"))

        return bboxes, mask_list
