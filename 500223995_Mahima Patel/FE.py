import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('CNN_FEmodel.h5')

# Define the labels
label_map = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}  # Update with your labels

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Streamlit app
st.title("Image Classification with CNN")
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = preprocess_image(img)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=1)[0]
    pred_label = label_map[pred_class]

    st.write(f"Prediction: {pred_label} ({prediction[0][pred_class] * 100:.2f}%)")
