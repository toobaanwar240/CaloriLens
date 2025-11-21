# nutrition_api/nutrition_mapper.py

def map_usda_to_basic_nutrition(food_data):
    """
    Converts USDA nutrients to a simplified set:
      - calories (kcal)
      - carbs_g
      - fat_g
      - protein_g
    """

    nutrients = food_data.get("foodNutrients", [])

    calories = protein = carbs = fat = 0.0

    for n in nutrients:
        name = n.get("nutrient", {}).get("name", "").lower()
        value = n.get("amount", 0.0)

        if "energy" in name and ("kcal" in name or "kilocalories" in name):
            calories = value
        elif "protein" in name:
            protein = value
        elif "carbohydrate" in name and "difference" in name:
            carbs = value
        elif "total lipid" in name or name == "total fat":
            fat = value

    return {
        "calories": float(calories),
        "protein_g": float(protein),
        "carbs_g": float(carbs),
        "fat_g": float(fat)
    }
