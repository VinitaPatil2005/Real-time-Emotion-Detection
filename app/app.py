import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_emotion
import subprocess
import sys
from pathlib import Path

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Real-Time Emotion Detection",
    layout="centered"
)

st.title("Real-Time Emotion Detection")

st.write(
    "Detect human emotions using a CNN model."
)

st.divider()

# ==========================================
# Sidebar
# ==========================================

option = st.sidebar.radio(
    "Choose Mode",
    ["Upload Image", "Webcam"]
)

# ==========================================
# Upload Image
# ==========================================

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        emotion, confidence = predict_emotion(image)

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width=450
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Emotion", emotion)

        with col2:
            st.metric("Confidence", f"{confidence:.2%}")

        st.progress(float(confidence))

# ==========================================
# Webcam
# ==========================================

else:

    st.subheader("Real-Time Webcam Detection")

    st.write(
        "Click the button below to launch the webcam."
    )

    if st.button("Start Webcam"):

        webcam_path = Path(__file__).parent / "webcam.py"

        subprocess.Popen(
            [sys.executable, str(webcam_path)]
        )

        st.success(
            "Webcam launched successfully!"
        )

st.divider()

st.caption(
    "Built using TensorFlow, OpenCV and Streamlit"
)