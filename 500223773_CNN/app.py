import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the saved model
model_path = "CNN.h5"
model = load_model(model_path)

# Load class labels
class_labels_path = "class_labels.pkl"
with open(class_labels_path, 'rb') as f:
    class_labels = pickle.load(f)

def preprocess_uploaded_image(uploaded_image):
    img_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2GRAY)

    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    if not np.any(img_blur):
        st.warning("Invalid image, please try another one")
        return None
    img_eq = cv2.equalizeHist(img_blur)

    img_resized = cv2.resize(img_eq, (256, 256))

    img_normalized = img_resized / 255.0

    img = np.reshape(img_normalized, (1, 256, 256, 1))
    return img

st.title('Facial Expression Recognition')

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = keras_image.load_img(uploaded_file, target_size=(256, 256), color_mode="rgb")
    img_array = keras_image.img_to_array(img)

    img_array = preprocess_uploaded_image(img_array)

    st.image(img, caption='Uploaded Image', use_column_width=True)

    if img_array is not None: 
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        st.subheader('Prediction:')
        st.write(f'Class: {class_labels[predicted_class]}')
        st.write(f'Confidence: {confidence:.2f}')

        st.subheader('Prediction Probabilities:')
        probs = {class_labels[i]: prediction[0][i] for i in range(len(class_labels))}
        st.write(probs)

        st.bar_chart(probs)
