# streamlit_app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from main_pipeline import analyze_food


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Food Nutrition Analyzer",
    page_icon="🍽",
    layout="wide"
)

st.title("🍽 AI Food Nutrition Analyzer")
st.markdown("Upload a food image and get **calories, macros, health score & recommendations** automatically.")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("---")

    if st.button("Analyze Food"):
        with st.spinner("Processing... Please wait."):

            # Store temporarily to pass path-like object
            temp_path = "temp_uploaded_image.jpg"
            cv2.imwrite(temp_path, img_bgr)

            result = analyze_food(temp_path)

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ----------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------
        food_name = result["food_name"]
        confidence = result["confidence"]
        grams = result["portion_grams"]
        nutrition = result["usda_nutrition"]
        llm = result["llm_report"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍔 Detected Food")
            st.write(f"**{food_name.title()}** (Confidence: {confidence*100:.1f}%)")
            st.write(f"Estimated Portion: **{grams:.1f} g**")

            st.subheader("🔥 USDA Nutrition")
            st.write(f"**Calories:** {nutrition['calories']:.1f} kcal")
            st.write(f"**Protein:** {nutrition['protein_g']:.1f} g")
            st.write(f"**Carbs:** {nutrition['carbs_g']:.1f} g")
            st.write(f"**Fat:** {nutrition['fat_g']:.1f} g")

        with col2:
            st.subheader("🧠 AI Nutrition Analysis (Groq)")
            st.write(f"### Health Score: **{llm['score']} / 100**")
            st.write("#### Summary")
            st.write(llm["summary"])

            st.write("#### Recommendations")
            for tip in llm["recommendations"]:
                st.write(f"- {tip}")

        st.success("Analysis complete!")


else:
    st.info("👆 Upload an image to begin.")
