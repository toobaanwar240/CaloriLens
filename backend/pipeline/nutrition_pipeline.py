import os
from groq import Groq

from CaloriLens.backend.nutrition_api.usda_client import USDANutritionClient
from CaloriLens.backend.llm_nutrition_agent.agent import NutritionLLMAgent


class NutritionPipeline:

    def __init__(self):
        usda_key = os.getenv("USDA_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not usda_key:
            raise ValueError("Missing USDA_API_KEY env variable.")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY env variable.")

        self.usda = USDANutritionClient(usda_key)
        self.groq_agent = NutritionLLMAgent(Groq(api_key=groq_key))

    def run(self, food_name: str, confidence: float, portion_grams: float):

        usda_result = self.usda.get_nutrition(food_name, portion_grams)

        # If USDA error → return early (no LLM needed)
        if "error" in usda_result:
            return {
                "food_name": food_name,
                "confidence": confidence,
                "portion_grams": portion_grams,
                "usda_error": usda_result["error"],
                "llm_report": None
            }

        # Step 2 — LLM Analysis
        llm_report = self.groq_agent.analyze_meal(usda_result)

        return {
            "food_name": food_name,
            "confidence": confidence,
            "portion_grams": portion_grams,
            "usda_nutrition": {
                "calories": usda_result["calories"],
                "protein_g": usda_result["protein_g"],
                "carbs_g": usda_result["carbs_g"],
                "fat_g": usda_result["fat_g"],
            },
            "llm_report": llm_report
        }
