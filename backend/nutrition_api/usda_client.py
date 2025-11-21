# nutrition_api/usda_client.py
#you need You need a USDA API key (free):
# https://fdc.nal.usda.gov/api-key-signup.html
import requests
from typing import Dict, Any, Optional, List
from .nutrition_mapper import map_usda_to_basic_nutrition


class USDANutritionClient:

    BASE_URL = "https://api.nal.usda.gov/fdc/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    # -------------------------------------------------------
    # 1. Search Food Endpoint
    # -------------------------------------------------------
    def search_food(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Returns USDA food search results.
        """

        url = f"{self.BASE_URL}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": max_results
        }

        resp = requests.get(url, params=params)
        data = resp.json()

        return data.get("foods", [])

    # -------------------------------------------------------
    # 2. Get Food Nutrition from FDC ID
    # -------------------------------------------------------
    def get_food_nutrition(self, fdc_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed USDA nutrition for a single item.
        """

        url = f"{self.BASE_URL}/food/{fdc_id}"
        params = {"api_key": self.api_key}

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            return None

        return resp.json()

    # -------------------------------------------------------
    # 3. Nutrition by food name + grams
    # -------------------------------------------------------
    def get_nutrition(self, food_name: str, grams: float) -> Dict:
        """
        End-to-end:
        - search food
        - pick best match
        - get full nutrients
        - convert to calories, carbs, fat, protein
        - scale by estimated grams
        """

        # 1) search best match
        results = self.search_food(food_name)
        if not results:
            return {"error": "Food not found in USDA database"}

        best_match = results[0]
        fdc_id = best_match["fdcId"]

        # 2) fetch full nutrient details
        food_data = self.get_food_nutrition(fdc_id)
        if not food_data:
            return {"error": "Failed to fetch nutrition data"}

        # 3) extract basic macros (per 100g)
        nutrition = map_usda_to_basic_nutrition(food_data)

        # 4) scale nutrients by portion weight (grams)
        factor = grams / 100.0

        scaled = {
            "food_name": food_name,
            "grams": grams,
            "calories": nutrition["calories"] * factor,
            "protein_g": nutrition["protein_g"] * factor,
            "carbs_g": nutrition["carbs_g"] * factor,
            "fat_g": nutrition["fat_g"] * factor,
        }

        return scaled
