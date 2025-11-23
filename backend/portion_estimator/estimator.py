import numpy as np
from typing import Dict, Optional

class PortionSizeEstimator:
    """
    Fixed portion size estimator with better calibration
    """
    
    def __init__(self, pixel_to_cm_ratio: float = 0.05):
        """
        Args:
            pixel_to_cm_ratio: Calibrated ratio (0.05 = 20px per cm, better for typical food photos)
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        
        # Typical serving sizes (grams)
        self.typical_portions = {
            # Pies & Baked Goods
            'apple_pie': 200,
            'pie': 200,
            'pizza': 250,
            
            # Main dishes
            'burger': 220,
            'hamburger': 220,
            'pasta': 200,
            'spaghetti_bolognese': 280,
            'lasagna': 300,
            
            # Rice dishes
            'rice': 180,
            'biryani': 300,
            'fried_rice': 250,
            
            # Asian foods
            'sushi': 120,
            'ramen': 400,
            'noodles': 200,
            'dumplings': 100,
            
            # Proteins
            'steak': 200,
            'chicken': 150,
            'fish': 150,
            
            # Sandwiches
            'sandwich': 180,
            'burger': 220,
            
            # Salads
            'salad': 100,
            'caesar_salad': 150,
            
            # Breakfast
            'pancakes': 150,
            'french_toast': 150,
            'omelette': 180,
            
            # Desserts
            'ice_cream': 100,
            'cake': 120,
            'cheesecake': 120,
        }
        
        # Food thickness (cm) - CRITICAL for flat foods
        self.thickness_map = {
            # Pies & Baked Goods
            'apple_pie': 5.0,      # Typical pie slice thickness
            'pie': 5.0,
            'pizza': 1.5,
            'pancakes': 1.2,
            'cake': 8.0,
            'cheesecake': 7.0,
            
            # Flat foods
            'dosa': 0.3,
            'tortilla': 0.2,
            'quesadilla': 1.5,
            'omelette': 2.0,
            'french_toast': 2.5,
            
            # Standard defaults
            'burger': 8.0,
            'sandwich': 6.0,
            'steak': 3.0,
        }
        
        # Food density (g/cm³) - CRITICAL for weight calculation
        self.density_map = {
            # Pies & Baked Goods (dense!)
            'apple_pie': 0.65,
            'pie': 0.65,
            'pizza': 0.55,
            'cake': 0.50,
            'cheesecake': 0.80,
            
            # Breads & Grains
            'pasta': 1.1,
            'rice': 0.85,
            'bread': 0.25,
            'noodles': 0.9,
            
            # Proteins (dense!)
            'chicken': 1.05,
            'beef': 1.05,
            'steak': 1.10,
            'fish': 1.05,
            'burger': 0.90,
            
            # Fried Foods (light)
            'french_fries': 0.40,
            'samosa': 0.70,
            
            # Salads (very light)
            'salad': 0.20,
            
            # Others
            'sushi': 0.75,
            'soup': 1.00,
            'ice_cream': 0.55,
        }
        
        # Depth ratio for volumetric estimation
        self.depth_ratio_map = {
            'burger': 0.6,
            'sandwich': 0.5,
            'steak': 0.3,
            'chicken': 0.4,
            'pie': 0.8,          # Pies are quite thick!
            'cake': 0.9,
            'sushi': 0.3,
        }
    
    def estimate(self, mask: np.ndarray, bbox: tuple, food_label: str) -> Dict:
        """
        Estimate portion weight with better calibration
        """
        # Normalize food label
        food_label_clean = food_label.lower().replace(' ', '_')
        
        # Calculate pixel area
        pixel_area = np.sum(mask)
        
        # Debug: Print raw measurements
        print(f"  [DEBUG] Pixel area: {pixel_area:,} pixels")
        
        # Convert to real-world area
        area_cm2 = pixel_area * (self.pixel_to_cm_ratio ** 2)
        print(f"  [DEBUG] Area: {area_cm2:.1f} cm²")
        
        # Get bbox dimensions
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        width_cm = width_px * self.pixel_to_cm_ratio
        height_cm = height_px * self.pixel_to_cm_ratio
        
        print(f"  [DEBUG] Dimensions: {width_cm:.1f} cm × {height_cm:.1f} cm")
        
        # Get thickness (CRITICAL!)
        thickness_cm = self.thickness_map.get(food_label_clean, height_cm * 0.4)
        print(f"  [DEBUG] Thickness: {thickness_cm:.1f} cm")
        
        # Calculate volume
        volume_cm3 = area_cm2 * thickness_cm
        print(f"  [DEBUG] Volume: {volume_cm3:.1f} cm³")
        
        # Get density (CRITICAL!)
        density = self.density_map.get(food_label_clean, 0.80)
        print(f"  [DEBUG] Density: {density:.2f} g/cm³")
        
        # Calculate weight from volume
        area_weight = volume_cm3 * density
        
        # Volume-based estimation (for 3D foods)
        depth_ratio = self.depth_ratio_map.get(food_label_clean, 0.4)
        depth_cm = height_cm * depth_ratio
        volume_3d = (4/3) * np.pi * (width_cm/2) * (height_cm/2) * (depth_cm/2) * 0.6
        volume_weight = volume_3d * density
        
        # Typical portion (with scaling)
        base_portion = self.typical_portions.get(food_label_clean, 200)
        bbox_area = width_px * height_px
        image_area = mask.shape[0] * mask.shape[1]
        fill_ratio = bbox_area / image_area
        
        if fill_ratio > 0.6:
            scale_factor = 1.4
        elif fill_ratio > 0.4:
            scale_factor = 1.0
        elif fill_ratio > 0.2:
            scale_factor = 0.75
        else:
            scale_factor = 0.5
        
        typical_weight = base_portion * scale_factor
        
        # Combine estimates intelligently
        final_weight, method = self._combine_estimates(
            area_weight, volume_weight, typical_weight, food_label_clean
        )
        
        print(f"  [DEBUG] Estimates: area={area_weight:.1f}g, volume={volume_weight:.1f}g, typical={typical_weight:.1f}g")
        print(f"  [DEBUG] Final: {final_weight:.1f}g ({method})")
        
        confidence = self._calculate_confidence(food_label_clean, method)
        
        return {
            "area_cm2": float(area_cm2),
            "volume_cm3": float(volume_cm3),
            "weight_grams": float(final_weight),
            "estimation_method": method,
            "confidence": confidence,
            "individual_estimates": {
                "area_based": float(area_weight),
                "volume_based": float(volume_weight),
                "typical_portion": float(typical_weight)
            }
        }
    
    def _combine_estimates(self, area_weight: float, volume_weight: float, 
                          typical_weight: float, food_label: str) -> tuple:
        """Combine estimates based on food type"""
        
        # Flat foods - use area-based
        flat_foods = ['pizza', 'pancakes', 'dosa', 'quesadilla', 'omelette', 'pie', 'apple_pie']
        
        # 3D/thick foods - use volume-based
        volumetric_foods = ['burger', 'sandwich', 'steak', 'chicken', 'cake', 'cheesecake']
        
        # Standard portions - use typical
        standard_foods = ['sushi', 'samosa', 'hot_dog', 'donut']
        
        if food_label in flat_foods:
            weight = area_weight * 0.7 + volume_weight * 0.1 + typical_weight * 0.2
            method = "area_based"
        elif food_label in volumetric_foods:
            weight = area_weight * 0.2 + volume_weight * 0.6 + typical_weight * 0.2
            method = "volume_based"
        elif food_label in standard_foods:
            weight = area_weight * 0.2 + volume_weight * 0.2 + typical_weight * 0.6
            method = "typical_portion"
        else:
            weight = area_weight * 0.3 + volume_weight * 0.3 + typical_weight * 0.4
            method = "combined"
        
        # Sanity check with wider range
        weight = max(30, min(weight, 2000))
        
        return weight, method
    
    def _calculate_confidence(self, food_label: str, method: str) -> float:
        """Calculate confidence score"""
        base_confidence = 0.70
        
        if food_label in self.typical_portions:
            base_confidence += 0.10
        
        if method == "typical_portion":
            base_confidence += 0.05
        
        high_confidence_foods = ['pizza', 'burger', 'steak', 'chicken', 'sushi', 'pie']
        if food_label in high_confidence_foods:
            base_confidence += 0.10
        
        return min(base_confidence, 0.95)


# Calibration helper
def calibrate_ratio(known_width_cm: float, pixel_width: int) -> float:
    """
    Calculate pixel_to_cm_ratio from a known reference
    
    Example:
        If you have a plate that's 26cm wide and it's 520 pixels in the image:
        ratio = calibrate_ratio(26, 520) = 0.05
    """
    return known_width_cm / pixel_width


if __name__ == "__main__":
    print("\n📏 PIXEL-TO-CM CALIBRATION EXAMPLES:")
    print("="*60)
    print("Common smartphone photo scenarios:")
    print(f"  - Plate (26cm) at 40cm distance → {calibrate_ratio(26, 520):.3f}")
    print(f"  - Close-up (15cm wide) → {calibrate_ratio(15, 600):.3f}")
    print(f"  - Full table shot (60cm) → {calibrate_ratio(60, 800):.3f}")
    print("\n💡 Recommended: Use 0.04-0.06 for typical food photos")
    print("="*60)