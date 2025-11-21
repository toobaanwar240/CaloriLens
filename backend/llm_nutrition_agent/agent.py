# llm_nutrition_agent/agent.py
#install groq 
#get groq key
import json
from typing import Dict, Any


class NutritionLLMAgent:
    """
    LLM Nutrition Agent using GROQ for extremely fast analysis.
    """

    def __init__(self, groq_client, model: str = "llama3-70b-8192"):
        """
        groq_client: Groq client instance
        model: Llama3 / Mixtral / Gemma (Groq models)
        """
        self.client = groq_client
        self.model = model

    # ----------------------------------------------------------
    # PROMPT BUILDING
    # ----------------------------------------------------------
    def _build_prompt(self, nutrition: Dict[str, Any]) -> str:

        food = nutrition["food_name"]
        grams = nutrition["grams"]
        cal = nutrition["calories"]
        protein = nutrition["protein_g"]
        carbs = nutrition["carbs_g"]
        fat = nutrition["fat_g"]

        return f"""
You are an expert nutrition analysis AI.

A food item was scanned: **{food}**, weighing **{grams:.1f} grams**.

Here are the nutrient values:
- Calories: {cal:.1f} kcal
- Protein: {protein:.1f} g
- Carbs: {carbs:.1f} g
- Fat:  {fat:.1f} g

Your tasks:
1. Give a nutrition **health score (0-100)**
2. Provide a **1-2 sentence summary**
3. Provide **3 short recommendations** to improve the meal
4. Output **ONLY JSON** in this exact structure:

{{
  "score": <number>,
  "summary": "<string>",
  "recommendations": [
    "<tip1>",
    "<tip2>",
    "<tip3>"
  ]
}}
"""
    # ----------------------------------------------------------
    # MAIN ANALYSIS FUNCTION (Groq)
    # ----------------------------------------------------------
    def analyze_meal(self, nutrition: Dict[str, Any]) -> Dict:

        prompt = self._build_prompt(nutrition)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional nutrition AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        text = response.choices[0].message["content"]

        # Try parsing JSON
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            print("⚠ Groq returned non-JSON output, returning fallback.")
            result = {
                "score": 50,
                "summary": "The AI could not parse nutrition correctly.",
                "recommendations": [
                    "Try scanning the food again.",
                    "Make sure the segmentation was accurate.",
                    "Ensure food name matches USDA database."
                ]
            }

        return result
