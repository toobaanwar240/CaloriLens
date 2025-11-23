import requests
from typing import Dict, Optional
import time

class USDANutritionClient:
    """
    USDA FoodData Central API Client - Updated for 2024 API
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # CORRECT BASE URL (v1 not v1/)
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        
        # Food name mapping
        self.food_mapping = {
            'pizza': 'pizza cheese',
            'burger': 'hamburger beef',
            'pasta': 'pasta cooked',
            'sushi': 'sushi roll',
            'ramen': 'ramen noodles',
            'steak': 'beef steak',
            'ice_cream': 'ice cream vanilla',
            'french_fries': 'french fries',
            'samosa': 'samosa',
            'grilled_chicken': 'chicken breast grilled',
            'salad': 'salad mixed',
            'caesar_salad': 'salad caesar',
            'chicken_curry': 'chicken curry',
            'chicken_quesadilla': 'quesadilla chicken',
            'fried_rice': 'rice fried',
            'biryani': 'rice pilaf',  # Closest match
            'dosa': 'crepe',
            'noodles': 'noodles cooked',
        }
    
    def get_nutrition(self, food_name: str, grams: float) -> Dict:
        """Get nutrition information for a food item"""
        try:
            # Normalize food name
            food_name_clean = food_name.lower().strip().replace(' ', '_')
            
            # Map to USDA-friendly search term
            search_term = self.food_mapping.get(food_name_clean, food_name)
            
            print(f"🔍 Searching USDA for: '{search_term}' (original: '{food_name}')")
            
            # Try mapped search first
            food_id = self._search_food(search_term)
            
            if not food_id:
                # Try original name
                print(f"   ⚠️ Not found with mapping, trying original name...")
                food_id = self._search_food(food_name)
            
            if not food_id:
                # Try generic fallback
                print(f"   ⚠️ Trying generic search...")
                generic_term = food_name.split('_')[0]  # Take first word
                food_id = self._search_food(generic_term)
            
            if not food_id:
                return self._get_fallback_nutrition(food_name, grams)
            
            # Get detailed nutrition
            nutrition_data = self._get_food_details(food_id, grams)
            
            return nutrition_data
            
        except Exception as e:
            print(f"❌ USDA API Error: {e}")
            return self._get_fallback_nutrition(food_name, grams)
    
    def _search_food(self, query: str) -> Optional[str]:
        """Search for food in USDA database"""
        try:
            url = f"{self.base_url}/foods/search"
            
            # FIXED: dataType as list, not repeated params
            params = {
                'query': query,
                'pageSize': 5,
                'api_key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            # Debug: print actual URL
            print(f"   📡 Request URL: {response.url}")
            
            response.raise_for_status()
            
            data = response.json()
            foods = data.get('foods', [])
            
            if not foods:
                print(f"   ⚠️ No results found for '{query}'")
                return None
            
            # Return first result
            best_match = foods[0]
            fdc_id = best_match.get('fdcId')
            description = best_match.get('description', 'Unknown')
            
            print(f"   ✓ Found: {description} (ID: {fdc_id})")
            
            return str(fdc_id)
            
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP Error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Search failed: {e}")
            return None
    
    def _get_food_details(self, fdc_id: str, grams: float) -> Dict:
        """Get detailed nutrition information"""
        try:
            url = f"{self.base_url}/food/{fdc_id}"
            params = {'api_key': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract nutrients
            nutrients = {}
            for nutrient in data.get('foodNutrients', []):
                nutrient_info = nutrient.get('nutrient', {})
                name = nutrient_info.get('name', '')
                amount = nutrient.get('amount', 0)
                nutrients[name] = amount
            
            # Get portion size (default 100g)
            portion_size = 100  # USDA typically uses 100g base
            scale = grams / portion_size
            
            # Extract key nutrients
            result = {
                "food_description": data.get('description', 'Unknown'),
                "fdc_id": fdc_id,
                "portion_grams": grams,
                "calories": self._get_nutrient(nutrients, ['Energy', 'Energy (Atwater General Factors)'], scale),
                "protein_g": self._get_nutrient(nutrients, ['Protein'], scale),
                "carbs_g": self._get_nutrient(nutrients, ['Carbohydrate, by difference', 'Total carbohydrate'], scale),
                "fat_g": self._get_nutrient(nutrients, ['Total lipid (fat)', 'Fat'], scale),
                "fiber_g": self._get_nutrient(nutrients, ['Fiber, total dietary'], scale),
                "sugar_g": self._get_nutrient(nutrients, ['Sugars, total including NLEA', 'Sugars, Total'], scale),
                "sodium_mg": self._get_nutrient(nutrients, ['Sodium, Na'], scale),
                "source": "USDA FoodData Central"
            }
            
            print(f"   ✓ Nutrition: {result['calories']:.0f} kcal, {result['protein_g']:.1f}g protein")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Failed to get details: {e}")
            return self._get_fallback_nutrition("unknown", grams)
    
    def _get_nutrient(self, nutrients: Dict, possible_names: list, scale: float = 1.0) -> float:
        """Extract nutrient value"""
        for name in possible_names:
            if name in nutrients:
                value = nutrients[name]
                # Handle kcal vs kJ for energy
                if 'Energy' in name and value > 1000:
                    value = value / 4.184  # Convert kJ to kcal
                return value * scale
        return 0.0
    
    def _get_fallback_nutrition(self, food_name: str, grams: float) -> Dict:
        """Fallback nutrition estimates when USDA fails"""
        
        # Generic estimates per 100g
        fallback_db = {
            'pizza': {'cal': 266, 'pro': 11, 'carb': 33, 'fat': 10},
            'burger': {'cal': 295, 'pro': 17, 'carb': 28, 'fat': 14},
            'pasta': {'cal': 131, 'pro': 5, 'carb': 25, 'fat': 1},
            'sushi': {'cal': 143, 'pro': 6, 'carb': 21, 'fat': 4},
            'ramen': {'cal': 436, 'pro': 9, 'carb': 62, 'fat': 15},
            'steak': {'cal': 271, 'pro': 25, 'carb': 0, 'fat': 19},
            'ice_cream': {'cal': 207, 'pro': 3.5, 'carb': 24, 'fat': 11},
            'french_fries': {'cal': 312, 'pro': 3.4, 'carb': 41, 'fat': 15},
            'salad': {'cal': 50, 'pro': 2, 'carb': 8, 'fat': 1},
            'caesar_salad': {'cal': 184, 'pro': 5, 'carb': 8, 'fat': 15},
            'chicken_curry': {'cal': 180, 'pro': 14, 'carb': 8, 'fat': 11},
            'chicken_quesadilla': {'cal': 218, 'pro': 10, 'carb': 22, 'fat': 10},
            'fried_rice': {'cal': 163, 'pro': 3.5, 'carb': 28, 'fat': 3.5},
            'biryani': {'cal': 170, 'pro': 7, 'carb': 25, 'fat': 5},
        }
        
        food_key = food_name.lower().replace(' ', '_')
        scale = grams / 100
        
        if food_key in fallback_db:
            data = fallback_db[food_key]
            print(f"   ✓ Using fallback estimates for {food_name}")
            
            return {
                "food_description": f"{food_name.title()} (estimated)",
                "fdc_id": "fallback",
                "portion_grams": grams,
                "calories": data['cal'] * scale,
                "protein_g": data['pro'] * scale,
                "carbs_g": data['carb'] * scale,
                "fat_g": data['fat'] * scale,
                "fiber_g": 2 * scale,
                "sugar_g": 3 * scale,
                "sodium_mg": 300 * scale,
                "source": "Estimated (USDA unavailable)",
                "is_estimate": True
            }
        
        # Generic fallback
        print(f"   ⚠️ No data available for {food_name}, using generic estimates")
        return {
            "food_description": f"{food_name.title()} (generic estimate)",
            "fdc_id": "generic",
            "portion_grams": grams,
            "calories": 200 * scale,
            "protein_g": 10 * scale,
            "carbs_g": 25 * scale,
            "fat_g": 8 * scale,
            "fiber_g": 2 * scale,
            "sugar_g": 3 * scale,
            "sodium_mg": 300 * scale,
            "source": "Generic Estimate",
            "is_estimate": True
        }


# Test the client
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("USDA_API_KEY")
    if not api_key:
        print("❌ USDA_API_KEY not found")
        exit(1)
    
    client = USDANutritionClient(api_key)
    
    # Test foods
    test_cases = [
        ('caesar_salad', 150),
        ('pizza', 250),
        ('burger', 200),
        ('chicken_curry', 300)
    ]
    
    for food, grams in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {food} ({grams}g)")
        print('='*60)
        
        result = client.get_nutrition(food, grams)
        
        print(f"\n✅ Result:")
        print(f"   {result['food_description']}")
        print(f"   Calories: {result['calories']:.0f} kcal")
        print(f"   Protein: {result['protein_g']:.1f}g")
        print(f"   Carbs: {result['carbs_g']:.1f}g")
        print(f"   Fat: {result['fat_g']:.1f}g")
        print(f"   Source: {result.get('source', 'Unknown')}")
        
        if result.get('is_estimate'):
            print(f"   ⚠️ Using estimates (USDA data unavailable)")