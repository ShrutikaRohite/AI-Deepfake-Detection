import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from data_generator import create_data_generators  # your generator

# --- Config ---
MODEL_PATH = "models/deepfake_model.h5"
TRAIN_DIR = "data/final_splits/train"
VAL_DIR   = "data/final_splits/val"
TEST_DIR  = "data/final_splits/test"
IMG_SIZE = 224
BATCH_SIZE = 16

# --- Load model (safe check) ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Train or place your model there.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# --- UI ---
st.title("DeepFake Detection App")

st.subheader("Model evaluation on test set (optional)")
# get generators (this also confirms your class mapping)
try:
    _, _, test_data = create_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
except Exception as e:
    st.error(f"Failed to create data generators: {e}")
    st.stop()

# show class mapping so you know which numeric label is which folder
st.write("**Folder → numeric label mapping (class_indices):**")
st.write(test_data.class_indices)

# derive names from class_indices: index->name
idx_to_name = {v: k for k, v in test_data.class_indices.items()}
positive_class_name = idx_to_name.get(1, None)   # the class that corresponds to model output ~1
negative_class_name = idx_to_name.get(0, None)

st.write(f"Interpreting model output `> threshold` as: **{positive_class_name}** (if 1 exists).")
st.write(f"Interpreting model output `<= threshold` as: **{negative_class_name}**.")

# show test accuracy quickly
try:
    loss, acc = model.evaluate(test_data, verbose=0)
    st.write(f"**Model Test Accuracy:** {acc * 100:.2f}%")
except Exception as e:
    st.write("Could not evaluate test set here (maybe large). You can run debug_predict.py for full metrics.")

# threshold slider and invert toggle
threshold = st.slider("Set prediction threshold", 0.0, 1.0, 0.5, 0.01)

# Image upload & prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction
    pred = float(model.predict(img_array)[0][0])

    # Decide label
    if positive_class_name is not None and negative_class_name is not None:
        assigned_name = positive_class_name if pred > threshold else negative_class_name
        label = "Fake" if assigned_name.lower().startswith("fake") else ("Real" if assigned_name.lower().startswith("real") else assigned_name)
    else:
        label = "Fake" if pred <= threshold else "Real"

    # Confidence
    confidence = pred * 100 if pred > threshold else (1 - pred) * 100

    # Show uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ✅ This stays inside the block
    if label.lower() == "real":
        st.markdown(f"<h3 style='color:green;'>✅ Prediction: {label} ({confidence:.2f}% confident)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>❌ Prediction: {label} ({confidence:.2f}% confident)</h3>", unsafe_allow_html=True)

    