import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import base64

# Function to convert image to base64
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
image_path = "bg.jpg"  # Update this to your local image path

# Get the base64 string of the image
base64_image = get_base64_image(image_path)

# Add custom CSS to set the background image and text color
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

# Load the trained model
MODEL_PATH = "resnet50_50_epoch.h5"  # Ensure this file is in the same directory
model = load_model(MODEL_PATH)
CATEGORIES = ["Normal", "Glaucoma"]  # Class labels
IMG_SIZE = 256

# Function for CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Streamlit UI
st.title("🔍 Glaucoma Detection from Retinal Images")
st.divider() 
st.write("Upload an image to classify it as Normal or Glaucoma.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded image to a format compatible with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_clahe(img)  # Apply CLAHE preprocessing
    
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image", width=200)
    with st.spinner("Processing..."):
        # Preprocess the image for the model
        img_array = np.expand_dims(img, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Apply ResNet50 preprocessing
        
        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        if CATEGORIES[predicted_class]=="Normal":
            st.success(f"### 🩺 Prediction: You have a **Healthy** eye")
            st.balloons()
        else:
            st.error(f"### ⚠️ Prediction: You may have **Glaucoma**. Please consult an ophthalmologist ASAP.  ")
        st.write(f"🧪 Confidence: **{confidence:.2f}%**")