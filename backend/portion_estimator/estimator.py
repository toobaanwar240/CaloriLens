# portion_estimator/estimator.py

import numpy as np
from typing import Dict
from .food_density import FOOD_DENSITY


class PortionSizeEstimator:
    """
    Estimates portion weight (grams) from segmentation mask + bbox.
    """

    def __init__(self, pixel_to_cm_ratio: float = 0.026):
        """
        pixel_to_cm_ratio: How many centimeters per pixel in your images
        Default ~0.026 cm/pixel (approx smartphone camera at ~40cm distance)
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio

    def estimate(self, mask: np.ndarray, bbox: tuple, food_label: str) -> Dict:
        """
        mask: (H, W) binary mask of food
        bbox: (x1, y1, x2, y2)
        food_label: predicted food name
        
        Returns:
            {
                "area_cm2": float,
                "volume_cm3": float,
                "weight_grams": float
            }
        """

        # --- 1. Calculate pixel area from mask ---
        pixel_area = np.sum(mask > 0)     # px²

        # Convert pixel area → cm²
        area_cm2 = pixel_area * (self.pixel_to_cm_ratio ** 2)

        # --- 2. Height estimation from bounding box ---
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        height_cm = pixel_height * self.pixel_to_cm_ratio

        # Assume thickness = 30% of height (approx food thickness)
        thickness_cm = height_cm * 0.30

        # --- 3. Volume = surface area × thickness ---
        volume_cm3 = area_cm2 * thickness_cm

        # --- 4. Lookup density for conversion to grams ---
        density = FOOD_DENSITY.get(food_label, 0.80)  # default density

        weight_grams = volume_cm3 * density

        return {
            "area_cm2": float(area_cm2),
            "volume_cm3": float(volume_cm3),
            "weight_grams": float(weight_grams)
        }
