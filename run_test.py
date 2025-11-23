import os
from CaloriLens.backend.pipeline.nutrition_pipeline import NutritionPipeline

# Ensure keys are loaded
from dotenv import load_dotenv
load_dotenv()

pipeline = NutritionPipeline()

result = pipeline.run(
    food_name="apple",
    confidence=0.93,
    portion_grams=145
)

print("\nFinal Pipeline Output:\n")
print(result)
