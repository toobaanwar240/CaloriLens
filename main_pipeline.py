# main_pipeline.py

import cv2
from groq import Groq
from huggingface_hub import hf_hub_download

# --- MODULE IMPORTS ---
from backend.classification.classifier import SmartEnsembleClassifier
from image_input import read_image
from backend.segmentation import SegmentationModel
from backend.portion_estimator.estimator import PortionSizeEstimator
from backend.nutrition_api.usda_client import USDANutritionClient
from backend.llm_nutrition_agent.agent import NutritionLLMAgent


# -------------------------------------------------------------
# INITIALIZE ALL COMPONENTS
# -------------------------------------------------------------


seg_model = SegmentationModel(
    model_type="yolo",          # change to "sam" anytime
    model_path="yolov8n-seg-food.pt",
    sam_checkpoint="sam_vit_h_4b8939.pth"
)

# Classification (ConvNeXt)
model1_path = hf_hub_download(
    repo_id="Tooba240/calorielens-food-classifier",
    filename="food_classifier_finetuned (1).pth"
)
model2_path = hf_hub_download(
    repo_id="Tooba240/calorielens-food-classifier",
    filename="food_classifier_finetuned (3).pth"
)
# Initialize
classifier = SmartEnsembleClassifier(
    model1_weights=model1_path,
    model2_weights=model2_path
)


# Portion Estimation
estimator = PortionSizeEstimator(pixel_to_cm_ratio=0.026)

# USDA API
usda = USDANutritionClient(api_key="USDA_API_KEY_HERE")

# Groq
groq_client = Groq(api_key="GROQ_API_KEY_HERE")
llm = NutritionLLMAgent(groq_client)


# -------------------------------------------------------------
# FULL PIPELINE FUNCTION
# -------------------------------------------------------------
def analyze_food(image_path: str):
    print("🔹 Loading image...")
    img = read_image(image_path)

    # --- SEGMENTATION ---
    print("🔹 Segmenting food...")
    bboxes, masks = seg_model.segment(img)

    if len(bboxes) == 0:
        return {"error": "No food detected in image."}

    # Use first detection (extend later for multiple foods)
    bbox = bboxes[0]

    x1, y1, x2, y2 = bbox[:4]
    crop = img[y1:y2, x1:x2]

    # --- CLASSIFICATION ---
    print("🔹 Classifying food...")
    cls = classifier.predict(crop)
    food_name = cls["label"]
    confidence = cls["confidence"]

    # --- PORTION SIZE ---
    print("🔹 Estimating portion size...")
    grams = portion.estimate(mask=masks[0])

    # --- USDA NUTRITION LOOKUP ---
    print("🔹 Fetching nutrition from USDA...")
    nutrition = usda.get_nutrition(food_name, grams)

    # Add food info for LLM agent
    nutrition["food_name"] = food_name
    nutrition["grams"] = grams

    # --- LLM GROQ ANALYSIS ---
    print("🔹 Generating LLM analysis...")
    final_report = llm.analyze_meal(nutrition)

    # Full structured return
    return {
        "food_name": food_name,
        "confidence": confidence,
        "portion_grams": grams,
        "usda_nutrition": nutrition,
        "llm_report": final_report
    }


# -------------------------------------------------------------
# RUN STANDALONE TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    result = analyze_food("sample_food.jpg")

    print("\n===== FINAL OUTPUT =====")
    print(result)
    print("========================")