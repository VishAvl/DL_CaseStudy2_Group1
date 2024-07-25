import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the VAE class
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)  # Cast inputs to float16
        z_mean, z_log_var = self.encoder(inputs)
        z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float16)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

# Load the encoder and decoder models
encoder = load_model('vae_encoder.h5', compile=False)
decoder = load_model('vae_decoder.h5', compile=False)

# Initialize the VAE model with the encoder and decoder
vae = VAE(encoder, decoder)

# Streamlit app
st.title("VAE Image Reconstruction")
st.write("Upload an image and see its reconstruction")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to the expected input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=(0, -1)).astype(np.float16)  # Add batch and channel dimensions and cast to float16

    # Get the reconstructed image from the VAE
    reconstructed, _, _ = vae(image_array)

    # Convert tensors to numpy arrays
    original_image = image_array[0, :, :, 0].astype(np.float32)  # Cast back to float32 for visualization
    reconstructed_image = reconstructed.numpy()[0, :, :, 0].astype(np.float32)  # Cast back to float32 for visualization

    # Display original and reconstructed images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    st.pyplot(fig)
