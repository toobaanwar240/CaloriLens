# 🥗 CaloriLens

AI-powered food recognition, portion estimation, and calorie tracking from images.

## Overview

CaloriLens analyzes food images to identify items, estimate portions, and calculate nutrition using USDA FoodData Central.

**Stack:**
- YOLOv8m-Seg (segmentation)
- ConvNeXt (classification, fine-tuned on Food-101)
- Custom portion estimation
- LLM nutrition insights

## Features

- **Segmentation** — YOLOv8m-Seg isolates food regions, removes background
- **Classification** — ConvNeXt classifies into 101 food categories
- **Portion Estimation** — Mask area + geometry → weight (g/ml)
- **Nutrition Lookup** — USDA database integration, scaled by portion
- **LLM Summary** — Natural language insights + health recommendations

## Architecture
```
Image → YOLOv8m-Seg → ConvNeXt → Portion Est. → USDA Lookup → LLM Summary
```

## Installation
```bash
git clone https://github.com/your-username/CaloriLens.git
cd CaloriLens
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app/streamlit_app.py
```

## Pipeline

1. Upload image
2. Segment food regions (YOLOv8m-Seg)
3. Classify food (ConvNeXt)
4. Estimate portion from mask area
5. Fetch & scale USDA nutrition data
6. Generate LLM summary

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics
- Streamlit

See `requirements.txt` for details.
