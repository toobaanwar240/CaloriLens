# image_input.py
"""
Image Input Module
------------------
Supports:
- Local path
- URL
- Webcam capture
- Streamlit file upload
"""

import cv2
import os
import numpy as np


def read_image(path: str):
    """
    Read an image from:
    - Local path
    - URL (http/https)
    Returns: BGR numpy array (OpenCV format)
    """
    if path.lower().startswith("http://") or path.lower().startswith("https://"):
        import requests
        from io import BytesIO
        from PIL import Image

        resp = requests.get(path)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download image from URL: {path}")

        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")

    return img


def capture_from_camera(device_idx: int = 0):
    """
    Capture a single frame from a webcam.
    """
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("No frame captured from camera")

    return frame


def upload_image(file_bytes: bytes):
    """
    Load image from raw bytes.
    """
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def read_streamlit_upload(uploaded_file):
    """
    Reads an image uploaded via Streamlit's st.file_uploader.

    Example:
        uploaded = st.file_uploader("Upload", type=["jpg","png"])
        if uploaded:
            img = read_streamlit_upload(uploaded)

    Returns: BGR numpy array
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.read()

    return upload_image(file_bytes)

