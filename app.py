import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import base64

# Function to convert image to base64
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
image_path = "bg.jpg"

if os.path.exists(image_path):
    base64_image = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{base64_image});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

MODEL_PATH = "resnet50_50_epoch.h5"
CATEGORIES = ["Normal", "Glaucoma"]
IMG_SIZE = 256
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error(
        "Trained weights file `resnet50_50_epoch.h5` was not found. "
        "Place it in the project root before running predictions."
    )

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

st.title("🔍 Glaucoma Detection from Retinal Images")
st.divider()
st.write("Upload an image to classify it as Normal or Glaucoma.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if model is None:
        st.warning("Cannot run a prediction until the model weights file is available.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = apply_clahe(img)

        st.image(uploaded_file, caption="Uploaded Image", width=200)
        with st.spinner("Processing..."):
            img_array = np.expand_dims(img, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            if CATEGORIES[predicted_class] == "Normal":
                st.success("### 🩺 Prediction: You have a **Healthy** eye")
                st.balloons()
            else:
                st.error("### ⚠️ Prediction: You may have **Glaucoma**. Please consult an ophthalmologist ASAP.")
            st.write(f"🧪 Confidence: **{confidence:.2f}%**")
