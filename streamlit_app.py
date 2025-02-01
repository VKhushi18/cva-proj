import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from PIL import Image
import os

# Ensure models directory exists
models_dir = "../models"
if not os.path.isdir(models_dir):
    st.error("Models directory not found. Please ensure the models are available in '../models' directory.")
    model_paths = {}
else:
    model_paths = {m: f"{models_dir}/{m}" for m in os.listdir(models_dir)}


def load_selected_model(model_name):
    return load_model(model_paths[model_name])

def plot_architecture(model_name):
    model = load_selected_model(model_name)
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
    return "model_architecture.png"

# Page 1: Model Architectures
def model_architecture_page():
    st.title("CNN Architectures")
    if not model_paths:
        st.error("No models available.")
        return
    model_name = st.selectbox("Select Model", list(model_paths.keys()))
    if model_name:
        img_path = plot_architecture(model_name)
        st.image(img_path, caption=f"Architecture of {model_name}", use_column_width=True)

# Page 2: Benchmark Plots
def benchmark_page():
    st.title("Model Performance")
    history_file = "history.npy"
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title("Training vs Validation Accuracy")
        ax1.legend()
        
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title("Training vs Validation Loss")
        ax2.legend()
        
        st.pyplot(fig)
    else:
        st.error("Benchmark history file not found. Train a model first.")

# Page 3: Model Inference
def inference_page():
    st.title("Model Inference")
    if not model_paths:
        st.error("No models available for inference.")
        return
    model_name = st.selectbox("Select Model", list(model_paths.keys()))
    model = load_selected_model(model_name)
    class_names = ['Class1', 'Class2', 'Class3']  # Update with actual class names
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image.resize((256, 256))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)
        
        st.write(f"Prediction: {predicted_class} with {confidence}% confidence")

# Streamlit Page Navigation
page = st.sidebar.selectbox("Choose Page", ["Model Architecture", "Benchmark Plots", "Inference"])
if page == "Model Architecture":
    model_architecture_page()
elif page == "Benchmark Plots":
    benchmark_page()
elif page == "Inference":
    inference_page()
