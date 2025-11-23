import cv2
import os
from groq import Groq
from huggingface_hub import hf_hub_download

# --- MODULE IMPORTS ---
from backend.classification.classifier import SmartEnsembleClassifier
from image_input import read_image  # Fixed import path
from backend.segmentation import SegmentationModel  # Fixed import path
from backend.portion_estimator.estimator import PortionSizeEstimator
from backend.nutrition_api.usda_client import USDANutritionClient
from backend.llm_nutrition_agent.agent import NutritionLLMAgent


# -------------------------------------------------------------
# INITIALIZE ALL COMPONENTS
# -------------------------------------------------------------

print("🚀 Initializing CalorieLens Pipeline...")

# Segmentation (YOLOv8)
print("\n1️⃣ Loading Segmentation Model...")
seg_model = SegmentationModel(
    model_type="yolo",
    model_path="yolov8m-seg.pt"  # Removed invalid parameters
)

# Classification (ConvNeXt Ensemble)
print("\n2️⃣ Loading Classification Models...")
model1_path = hf_hub_download(
    repo_id="Tooba240/calorielens-food-classifier",
    filename="food_classifier_finetuned (1).pth"
)
model2_path = hf_hub_download(
    repo_id="Tooba240/calorielens-food-classifier",
    filename="food_classifier_finetuned (3).pth"
)
classifier = SmartEnsembleClassifier(
    model1_weights=model1_path,
    model2_weights=model2_path
)

# Portion Estimation
print("\n3️⃣ Initializing Portion Estimator...")
portion_estimator = PortionSizeEstimator(
    pixel_to_cm_ratio=0.05  # Fixed parameter name
)

# USDA API
print("\n4️⃣ Setting up USDA API Client...")
USDA_API_KEY = os.getenv("USDA_API_KEY")
if not USDA_API_KEY:
    print("⚠️  Warning: USDA_API_KEY not found in environment variables")
    print("   Set it with: export USDA_API_KEY='your_key_here'")
usda_client = USDANutritionClient(api_key=USDA_API_KEY)

# Groq LLM Agent
print("\n5️⃣ Setting up Groq LLM Agent...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  Warning: GROQ_API_KEY not found in environment variables")
    print("   Set it with: export GROQ_API_KEY='your_key_here'")
groq_client = Groq(api_key=GROQ_API_KEY)
llm_agent = NutritionLLMAgent(groq_client)

print("\n✅ All components loaded successfully!\n")


# -------------------------------------------------------------
# FULL PIPELINE FUNCTION
# -------------------------------------------------------------
def analyze_food(image_path: str):
    """
    Complete food analysis pipeline
    
    Args:
        image_path: Path to food image
        
    Returns:
        Dictionary with complete analysis results
    """
    print(f"🔹 Analyzing image: {image_path}")
    
    # --- 1. LOAD IMAGE ---
    print("🔹 Step 1/5: Loading image...")
    img = read_image(image_path)
    if img is None:
        return {"error": f"Failed to load image: {image_path}"}

    # --- 2. SEGMENTATION ---
    print("🔹 Step 2/5: Segmenting food...")
    bboxes, masks = seg_model.segment(img, conf_threshold=0.25)

    if len(bboxes) == 0:
        return {"error": "No food detected in image."}

    print(f"   ✓ Found {len(bboxes)} food item(s)")

    # Use largest detection (you can extend this for multiple foods later)
    bbox = bboxes[0]
    mask = masks[0]
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Crop food region
    crop = img[y1:y2, x1:x2].copy()

    # --- 3. CLASSIFICATION ---
    print("🔹 Step 3/5: Classifying food...")
    classification_result = classifier.predict(crop, top_k=3)
    
    food_name = classification_result["label"]
    confidence = classification_result["confidence"]
    
    print(f"   ✓ Predicted: {food_name} ({confidence:.1%} confidence)")
    print(f"   Model used: {classification_result.get('chosen_model', 'unknown')}")
    print(f"   Reasoning: {classification_result.get('reasoning', 'N/A')}")

    # --- 4. PORTION SIZE ESTIMATION ---
    print("🔹 Step 4/5: Estimating portion size...")
    portion_result = portion_estimator.estimate(
        mask=mask,
        bbox=(x1, y1, x2, y2),
        food_label=food_name
    )
    
    grams = portion_result["weight_grams"]
    print(f"   ✓ Estimated weight: {grams:.1f}g")
    print(f"   Area: {portion_result['area_cm2']:.1f} cm²")
    print(f"   Volume: {portion_result['volume_cm3']:.1f} cm³")

    # --- 5. USDA NUTRITION LOOKUP ---
    print("🔹 Step 5/5: Fetching nutrition data...")
    nutrition_data = usda_client.get_nutrition(food_name, grams)
    
    if "error" in nutrition_data:
        print(f"   ⚠️  USDA API error: {nutrition_data['error']}")
        return {
            "food_name": food_name,
            "confidence": confidence,
            "portion_grams": grams,
            "portion_details": portion_result,
            "classification": classification_result,
            "usda_error": nutrition_data["error"],
            "llm_report": None
        }
    
    print(f"   ✓ Nutrition data retrieved")
    print(f"   Calories: {nutrition_data.get('calories', 0):.0f} kcal")

    # Add food context for LLM
    nutrition_data["food_name"] = food_name
    nutrition_data["grams"] = grams

    # --- 6. LLM ANALYSIS ---
    print("🔹 Generating AI nutrition analysis...")
    llm_report = llm_agent.analyze_meal(nutrition_data)
    print("   ✓ AI analysis complete")

    # --- RETURN COMPLETE RESULTS ---
    return {
        "food_name": food_name,
        "confidence": confidence,
        "portion_grams": grams,
        "portion_details": portion_result,
        "classification_details": {
            "top_predictions": classification_result.get("top_k", []),
            "chosen_model": classification_result.get("chosen_model"),
            "reasoning": classification_result.get("reasoning")
        },
        "usda_nutrition": {
            "calories": nutrition_data.get("calories", 0),
            "protein_g": nutrition_data.get("protein_g", 0),
            "carbs_g": nutrition_data.get("carbs_g", 0),
            "fat_g": nutrition_data.get("fat_g", 0),
        },
        "llm_report": llm_report,
        "visualization": {
            "bbox": [x1, y1, x2, y2],
            "has_mask": True
        }
    }


# -------------------------------------------------------------
# RUN STANDALONE TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "pizza_salad.jpg"
        print(f"💡 No image specified, using default: {image_path}")
        print(f"   Usage: python main_pipeline.py <image_path>\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Run analysis
    print("="*60)
    result = analyze_food(image_path)
    print("="*60)

    # Display results
    print("\n" + "="*60)
    print("📊 FINAL ANALYSIS RESULTS")
    print("="*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"\n🍽️  Food: {result['food_name']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚖️  Portion: {result['portion_grams']:.1f}g")
        
        if "usda_nutrition" in result:
            print(f"\n📈 Nutrition Facts:")
            print(f"   Calories: {result['usda_nutrition']['calories']:.0f} kcal")
            print(f"   Protein: {result['usda_nutrition']['protein_g']:.1f}g")
            print(f"   Carbs: {result['usda_nutrition']['carbs_g']:.1f}g")
            print(f"   Fat: {result['usda_nutrition']['fat_g']:.1f}g")
        
        if result.get("llm_report"):
            print(f"\n🤖 AI Analysis:")
            print(f"{result['llm_report']}")
    
    print("="*60)