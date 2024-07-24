import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the VAE autoencoder model from the pickle file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to display images
def display_images(original, generated):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original.reshape(128, 128), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(generated.reshape(128, 128), cmap='gray')
    axes[1].set_title("Generated Image")
    axes[1].axis('off')
    
    st.pyplot(fig)

# Streamlit application
st.title("VAE Autoencoder Image Generation")

# Load the model
model_path = 'vae_autoencoder.pkl'
vae_autoencoder = load_model(model_path)
st.write("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Generate image using the VAE autoencoder
    generated_image = vae_autoencoder.predict(preprocessed_image)
    
    # Display the original and generated images
    display_images(preprocessed_image, generated_image)
