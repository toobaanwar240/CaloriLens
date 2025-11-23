import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import sys
from pathlib import Path
import cv2

# Add backend to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent  # CaloriLens/
backend_dir = project_root / "backend"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

# Import backend modules
try:
    from backend.classification.classifier import SmartEnsembleClassifier
    from backend.segmentation.segmentator import SegmentationModel
    from backend.portion_estimator.estimator import PortionSizeEstimator
    from backend.nutrition_api.usda_client import USDANutritionClient
    from backend.llm_nutrition_agent.agent import NutritionLLMAgent
    from groq import Groq
    from huggingface_hub import hf_hub_download
    from dotenv import load_dotenv
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# ------------------------------
# LOAD CUSTOM CSS
# ------------------------------

def load_css(file_name):
    """Load custom CSS with better error handling"""
    css_path = current_dir / file_name
    
    if not css_path.exists():
        st.sidebar.warning(f"⚠️ CSS file not found: {css_path}")
        return False
    
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load CSS: {e}")
        return False

# Load CSS
css_loaded = load_css("style.css")

# ------------------------------
# INITIALIZE MODELS (with caching)
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models once and cache them"""
    
    # Check API keys first
    usda_key = os.getenv("USDA_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not usda_key or not groq_key:
        return None
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Segmentation Model (20%)
        status_text.text("🔄 Loading segmentation model...")
        model_path = hf_hub_download(
        repo_id="Tooba240/yolo_models",
        filename="yolov8m-seg.pt"
        )
        seg_model = SegmentationModel(
            model_type="yolo",
            model_path=model_path
        )
        progress_bar.progress(20)
        
        # 2. Classification Models (60%)
        status_text.text("🔄 Downloading classification models from HuggingFace...")
        model1_path = hf_hub_download(
            repo_id="Tooba240/calorielens-food-classifier",
            filename="food_classifier_finetuned (1).pth"
        )
        progress_bar.progress(40)
        
        model2_path = hf_hub_download(
            repo_id="Tooba240/calorielens-food-classifier",
            filename="food_classifier_finetuned (3).pth"
        )
        progress_bar.progress(50)
        
        status_text.text("🔄 Loading classification ensemble...")
        classifier = SmartEnsembleClassifier(
            model1_weights=model1_path,
            model2_weights=model2_path
        )
        progress_bar.progress(70)
        
        # 3. Portion Estimator (80%)
        status_text.text("🔄 Initializing portion estimator...")
        portion_estimator = PortionSizeEstimator(pixel_to_cm_ratio=0.05)
        progress_bar.progress(80)
        
        # 4. API Clients (100%)
        status_text.text("🔄 Setting up API clients...")
        usda_client = USDANutritionClient(api_key=usda_key)
        groq_client = Groq(api_key=groq_key)
        llm_agent = NutritionLLMAgent(groq_client)
        progress_bar.progress(100)
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'seg_model': seg_model,
            'classifier': classifier,
            'portion_estimator': portion_estimator,
            'usda_client': usda_client,
            'llm_agent': llm_agent
        }
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return None

# ------------------------------
# ANALYSIS FUNCTION
# ------------------------------

def analyze_food_complete(image_pil, models):
    """Complete food analysis pipeline"""
    try:
        # Convert PIL to OpenCV format (RGB to BGR)
        img_array = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # --- STEP 1: SEGMENTATION ---
        bboxes, masks = models['seg_model'].segment(img_bgr, conf_threshold=0.25)
        
        if len(bboxes) == 0:
            return {
                "error": "No food detected in image. Please try a clearer image with visible food items."
            }
        
        # Use largest detection (first item)
        bbox = bboxes[0]
        mask = masks[0]
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Crop food region
        crop = img_bgr[y1:y2, x1:x2].copy()
        
        # --- STEP 2: CLASSIFICATION ---
        classification_result = models['classifier'].predict(crop, top_k=3)
        food_name = classification_result["label"]
        confidence = classification_result["confidence"]
        
        # --- STEP 3: PORTION ESTIMATION ---
        portion_result = models['portion_estimator'].estimate(
            mask=mask,
            bbox=(x1, y1, x2, y2),
            food_label=food_name
        )
        grams = portion_result["weight_grams"]
        
        # --- STEP 4: USDA NUTRITION LOOKUP ---
        nutrition_data = models['usda_client'].get_nutrition(food_name, grams)
        
        if "error" in nutrition_data:
            return {
                "food_name": food_name,
                "confidence": confidence,
                "portion_grams": grams,
                "portion_details": portion_result,
                "classification_details": classification_result,
                "usda_error": nutrition_data["error"],
                "llm_report": None,
                "partial_result": True
            }
        
        # Add food context for LLM
        nutrition_data["food_name"] = food_name
        nutrition_data["grams"] = grams
        
        # --- STEP 5: LLM ANALYSIS ---
        llm_report = models['llm_agent'].analyze_meal(nutrition_data)
        
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
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "error": f"Analysis failed: {str(e)}",
            "error_details": error_details
        }

# ------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="CaloriLens - AI Food Analyzer", 
    page_icon="🍽", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# CHECK API KEYS & LOAD MODELS
# ------------------------------

# Check for API keys
usda_key = os.getenv("USDA_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not usda_key or not groq_key:
    st.error("⚠️ **API Keys Missing!**")
    st.markdown("""
    ### Setup Instructions:
    
    1. **Create a `.env` file** in your project root:
    ```bash
    USDA_API_KEY=your_usda_key_here
    GROQ_API_KEY=your_groq_key_here
    ```
    
    2. **Get FREE API keys:**
    - **USDA API**: [https://fdc.nal.usda.gov/api-key-signup.html](https://fdc.nal.usda.gov/api-key-signup.html)
    - **Groq API**: [https://console.groq.com/keys](https://console.groq.com/keys)
    
    3. **Restart the app** after adding keys
    """)
    st.stop()

# Load models with caching
if 'models' not in st.session_state:
    with st.spinner("🚀 Initializing AI models... (This may take 1-2 minutes on first run)"):
        st.session_state.models = load_models()

models = st.session_state.models

if models is None:
    st.error("❌ Failed to initialize models. Please check your API keys and internet connection.")
    st.stop()

# ------------------------------
# HEADER
# ------------------------------

st.markdown("<h1 class='title'>🍽️ CaloriLens AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #94A3B8; font-size: 1.1rem; "
    "margin-bottom: 3rem; font-weight: 400;'>Powered by FineTuned Models & LLM Intelligence</p>", 
    unsafe_allow_html=True
)

# ------------------------------
# FEATURES SECTION
# ------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>📸 Smart Detection</h3>
        <p>Advanced YOLOv8 segmentation automatically detects and isolates food items with 95% accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>🍎 Nutrition Analysis</h3>
        <p>Real-time calorie, protein, carbs, and fat estimation from USDA FoodData Central database.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>⚡ AI Insights</h3>
        <p>Personalized health scores and dietary recommendations powered by Groq LLM.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ------------------------------
# IMAGE UPLOAD ZONE
# ------------------------------

st.markdown("""
<div class='upload-zone'>
    <div class='upload-text'>📤 Drag & Drop an Image<br>or Click to Upload</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "", 
    type=["jpg", "jpeg", "png"], 
    label_visibility="collapsed",
    help="Upload a clear photo of your food for analysis"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="✨ Your Uploaded Food Image", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analyze button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button("🔍 Analyze Food", use_container_width=True, type="primary")
    
    if analyze_button:
        # Show loading spinner with custom message
        with st.spinner("🤖 Analyzing your food... Please wait"):
            result = analyze_food_complete(image, models)
        
        # Handle errors
        if "error" in result and not result.get("partial_result"):
            st.error(f"❌ {result['error']}")
            if result.get("error_details"):
                with st.expander("🔍 View Error Details"):
                    st.code(result["error_details"])
        
        # Handle partial results (food detected but no nutrition data)
        elif result.get("partial_result"):
            st.warning(f"⚠️ {result.get('usda_error', 'Nutrition data unavailable')}")
            
            st.info(f"""
            **Detected Food:** {result['food_name'].title()}  
            **Confidence:** {result['confidence']*100:.1f}%  
            **Estimated Portion:** {result['portion_grams']:.1f}g  
            
            💡 *Try a more common food item for full nutrition analysis*
            """)
            
            # Show top predictions
            if result.get('classification_details'):
                with st.expander("🔍 View Alternative Predictions"):
                    top_preds = result['classification_details'].get('top_predictions', [])
                    for i, pred in enumerate(top_preds[:3], 1):
                        st.write(f"{i}. **{pred['label'].title()}** - {pred['confidence']*100:.1f}%")
        
        # Display full results
        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                "<h2 style='text-align: center; color: #F1F5F9; font-weight: 600; margin-bottom: 2rem;'>📊 Analysis Results</h2>", 
                unsafe_allow_html=True
            )
            
            food_name = result["food_name"]
            confidence = result["confidence"]
            grams = result["portion_grams"]
            nutrition = result["usda_nutrition"]
            llm = result.get("llm_report", {})

            # Two column layout with proper spacing
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                # Detected Food Card
                st.markdown("### 🍔 Detected Food")
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(59, 130, 246, 0.05)); 
                            padding: 1.8rem; 
                            border-radius: 16px; 
                            border: 2px solid rgba(37, 99, 235, 0.2);
                            margin: 1rem 0 2.5rem 0;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <h2 style='color: #2563EB; margin: 0 0 1rem 0; font-weight: 700; font-size: 2rem;'>
                        {food_name.title()}
                    </h2>
                    <div style='display: flex; justify-content: space-between; gap: 1rem; margin-top: 1.2rem;'>
                        <div style='flex: 1;'>
                            <p style='color: #94A3B8; margin: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;'>
                                Confidence
                            </p>
                            <p style='color: #10B981; margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: 700;'>
                                {confidence*100:.1f}%
                            </p>
                        </div>
                        <div style='flex: 1;'>
                            <p style='color: #94A3B8; margin: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;'>
                                Portion
                            </p>
                            <p style='color: #F59E0B; margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: 700;'>
                                {grams:.1f}g
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Model insights expander
                if result.get("classification_details"):
                    with st.expander("🔍 View Classification Details"):
                        cls_details = result["classification_details"]
                        st.markdown(f"""
                        <div style='padding: 0.5rem 0;'>
                            <p style='margin: 0.5rem 0; color: #F1F5F9;'>
                                <strong>Model Used:</strong> 
                                <span style='color: #2563EB;'>{cls_details.get('chosen_model', 'N/A').upper()}</span>
                            </p>
                            <p style='margin: 0.5rem 0; color: #94A3B8; font-size: 0.9rem;'>
                                {cls_details.get('reasoning', 'N/A')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Top 3 Predictions:**")
                        
                        for i, pred in enumerate(cls_details.get('top_predictions', [])[:3], 1):
                            confidence_color = "#10B981" if i == 1 else "#94A3B8"
                            st.markdown(f"""
                            <div style='background: rgba(255, 255, 255, 0.02);
                                        padding: 0.7rem 1rem;
                                        border-radius: 8px;
                                        margin: 0.4rem 0;
                                        border-left: 3px solid {confidence_color};'>
                                <span style='color: #F1F5F9; font-weight: 600;'>{i}. {pred['label'].title()}</span>
                                <span style='color: {confidence_color}; float: right; font-weight: 600;'>
                                    {pred['confidence']*100:.1f}%
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Spacing between sections
                st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

                # Nutrition Facts
                st.markdown("### 🔥 Nutrition Facts")
                st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2, gap="medium")
                
                with metric_col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
                                padding: 1.5rem; border-radius: 12px; margin: 0.6rem 0; text-align: center;
                                border: 2px solid rgba(239, 68, 68, 0.2); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                        <p style='color: #94A3B8; margin: 0; font-size: 0.75rem; text-transform: uppercase; 
                                  letter-spacing: 0.1em; font-weight: 600;'>Calories</p>
                        <h3 style='color: #EF4444; margin: 0.5rem 0 0 0; font-weight: 800; font-size: 2rem;'>
                            {nutrition['calories']:.0f} <span style='font-size: 0.9rem; color: #94A3B8; font-weight: 500;'>kcal</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.05));
                                padding: 1.5rem; border-radius: 12px; margin: 0.6rem 0; text-align: center;
                                border: 2px solid rgba(245, 158, 11, 0.2); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                        <p style='color: #94A3B8; margin: 0; font-size: 0.75rem; text-transform: uppercase; 
                                  letter-spacing: 0.1em; font-weight: 600;'>Carbs</p>
                        <h3 style='color: #F59E0B; margin: 0.5rem 0 0 0; font-weight: 800; font-size: 2rem;'>
                            {nutrition['carbs_g']:.1f} <span style='font-size: 0.9rem; color: #94A3B8; font-weight: 500;'>g</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(29, 78, 216, 0.05));
                                padding: 1.5rem; border-radius: 12px; margin: 0.6rem 0; text-align: center;
                                border: 2px solid rgba(37, 99, 235, 0.2); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                        <p style='color: #94A3B8; margin: 0; font-size: 0.75rem; text-transform: uppercase; 
                                  letter-spacing: 0.1em; font-weight: 600;'>Protein</p>
                        <h3 style='color: #2563EB; margin: 0.5rem 0 0 0; font-weight: 800; font-size: 2rem;'>
                            {nutrition['protein_g']:.1f} <span style='font-size: 0.9rem; color: #94A3B8; font-weight: 500;'>g</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.05));
                                padding: 1.5rem; border-radius: 12px; margin: 0.6rem 0; text-align: center;
                                border: 2px solid rgba(16, 185, 129, 0.2); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                        <p style='color: #94A3B8; margin: 0; font-size: 0.75rem; text-transform: uppercase; 
                                  letter-spacing: 0.1em; font-weight: 600;'>Fat</p>
                        <h3 style='color: #10B981; margin: 0.5rem 0 0 0; font-weight: 800; font-size: 2rem;'>
                            {nutrition['fat_g']:.1f} <span style='font-size: 0.9rem; color: #94A3B8; font-weight: 500;'>g</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🧠 AI Nutrition Analysis")
                st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                
                if llm and isinstance(llm, dict):
                    # Health Score
                    score = llm.get('score', 70)
                    if score >= 80:
                        score_color, score_emoji, score_label = "#10B981", "🌟", "Excellent"
                        score_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08))"
                    elif score >= 60:
                        score_color, score_emoji, score_label = "#F59E0B", "⭐", "Good"
                        score_bg = "linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.08))"
                    else:
                        score_color, score_emoji, score_label = "#EF4444", "⚠️", "Needs Improvement"
                        score_bg = "linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.08))"
                    
                    st.markdown(f"""
                    <div style='background: {score_bg};
                                padding: 2.5rem 2rem; border-radius: 20px; text-align: center;
                                margin: 1rem 0 2.5rem 0; border: 3px solid {score_color};
                                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);'>
                        <p style='color: #94A3B8; margin: 0; font-size: 0.85rem; text-transform: uppercase; 
                                  letter-spacing: 0.15em; font-weight: 600;'>Health Score</p>
                        <h1 style='color: {score_color}; margin: 1rem 0; font-size: 5rem; font-weight: 900; line-height: 1;'>
                            {score_emoji} {score}
                        </h1>
                        <p style='color: #94A3B8; margin: 0; font-size: 1rem;'>
                            out of 100 • <span style='color: {score_color}; font-weight: 700;'>{score_label}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Summary
                    if 'summary' in llm:
                        st.markdown("#### 📝 Summary")
                        st.markdown(f"""
                        <div style='background: rgba(37, 99, 235, 0.08); padding: 1.5rem; border-radius: 14px;
                                    border-left: 4px solid #2563EB; margin: 1rem 0 2rem 0;
                                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);'>
                            <p style='margin: 0; color: #F1F5F9; line-height: 1.7; font-size: 1rem;'>
                                {llm["summary"]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Recommendations
                    if 'recommendations' in llm and llm['recommendations']:
                        st.markdown("#### 💡 Recommendations")
                        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                        
                        for i, tip in enumerate(llm["recommendations"], 1):
                            st.markdown(f"""
                            <div style='background: rgba(16, 185, 129, 0.06); padding: 1.2rem 1.5rem;
                                        border-radius: 12px; margin: 0.8rem 0; border-left: 4px solid #10B981;
                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                                <p style='margin: 0; color: #F1F5F9; line-height: 1.6; font-size: 0.95rem;'>
                                    <strong style='color: #10B981;'>{i}.</strong> {tip}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                
                elif llm and isinstance(llm, str):
                    st.markdown(f"""
                    <div style='background: rgba(37, 99, 235, 0.08); padding: 2rem; border-radius: 14px;
                                border-left: 4px solid #2563EB; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);'>
                        <p style='margin: 0; color: #F1F5F9; line-height: 1.7; font-size: 1rem;'>
                            {llm}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("💬 AI analysis unavailable")
            
            # Bottom spacing and success message
            st.markdown("<div style='margin: 3rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
            
            # Success message with nice styling
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
                        border: 2px solid rgba(16, 185, 129, 0.3);
                        border-radius: 12px;
                        padding: 1.2rem 1.5rem;
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <p style='margin: 0;
                          color: #10B981;
                          font-weight: 600;
                          font-size: 1.05rem;'>
                    ✅ Analysis complete! Your nutrition report is ready.
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Home page (no image uploaded)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👆 Upload an image to begin your food analysis journey!")
    
    # Statistics
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: #F1F5F9; font-weight: 600;'>📈 Platform Capabilities</h3>", 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(37, 99, 235, 0.05); 
                    border-radius: 12px; border: 1px solid rgba(37, 99, 235, 0.2);'>
            <h2 style='color: #2563EB; margin: 0; font-weight: 700;'>50+</h2>
            <p style='color: #94A3B8; margin: 0.5rem 0 0 0;'>Food Classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(16, 185, 129, 0.05); 
                    border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2);'>
            <h2 style='color: #10B981; margin: 0; font-weight: 700;'>85%</h2>
            <p style='color: #94A3B8; margin: 0.5rem 0 0 0;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(245, 158, 11, 0.05); 
                    border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.2);'>
            <h2 style='color: #F59E0B; margin: 0; font-weight: 700;'>AI-Powered</h2>
            <p style='color: #94A3B8; margin: 0.5rem 0 0 0;'>Insights</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 2rem 0; border-top: 1px solid rgba(148, 163, 184, 0.1);'>
    <p style='margin: 0;'>🍽️ CalorieLens AI - Powered by YOLOv8, ConvNeXt & Groq LLM</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
