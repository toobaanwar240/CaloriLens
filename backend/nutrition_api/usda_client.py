# nutrition_api/usda_client.py

import os
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from .nutrition_mapper import map_usda_to_basic_nutrition

# Load .env if present
load_dotenv()


class USDANutritionClient:

    BASE_URL = "https://api.nal.usda.gov/fdc/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("USDA_API_KEY")
        if not self.api_key:
            raise ValueError("USDA API key missing. Set USDA_API_KEY in environment.")

    # -------------------------------------------------------
    # 1. Search Food
    # -------------------------------------------------------
    def search_food(self, query: str, max_results: int = 5) -> List[Dict]:
        url = f"{self.BASE_URL}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": max_results
        }

        resp = requests.get(url, params=params)

        # Debug safety
        if resp.status_code != 200:
            print("USDA search error:", resp.status_code, resp.text)
            return []

        try:
            data = resp.json()
        except Exception:
            print("USDA returned non-JSON response:", resp.text)
            return []

        return data.get("foods", [])

    # -------------------------------------------------------
    # 2. Get Food Nutrition
    # -------------------------------------------------------
    def get_food_nutrition(self, fdc_id: int) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/food/{fdc_id}"
        params = {"api_key": self.api_key}

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            print("USDA detail fetch failed:", resp.status_code, resp.text)
            return None

        try:
            return resp.json()
        except Exception:
            print("Invalid JSON from USDA:", resp.text)
            return None

    # -------------------------------------------------------
    # 3. Search → Fetch → Scale
    # -------------------------------------------------------
    def get_nutrition(self, food_name: str, grams: float) -> Dict:
        results = self.search_food(food_name)
        if not results:
            return {"error": "Food not found in USDA database"}

        best_match = results[0]
        fdc_id = best_match["fdcId"]

        food_data = self.get_food_nutrition(fdc_id)
        if not food_data:
            return {"error": "Failed to fetch nutrition data"}

        # Extract macros
        nutrition = map_usda_to_basic_nutrition(food_data)

        factor = grams / 100.0

        return {
            "food_name": food_name,
            "grams": grams,
            "calories": nutrition["calories"] * factor,
            "protein_g": nutrition["protein_g"] * factor,
            "carbs_g": nutrition["carbs_g"] * factor,
            "fat_g": nutrition["fat_g"] * factor,
        }
