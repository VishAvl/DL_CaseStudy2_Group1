import streamlit as st
import requests
import json
from io import BytesIO

st.title("Face Expression classifier")

img_file_buffer = st.file_uploader("Upload your expressive face")

class_labels = {
    0: 'Surprise',
    1: 'Sad',
    2: 'Ahegao',
    3: 'happy',
    4: 'Neutral',
    5: 'Angry'
}

if img_file_buffer is not None:
    img_bytes = img_file_buffer.getvalue()

    files = {
        'face_img': BytesIO(img_bytes)
    }

    URL = "https://gruhit-patel-face-expression-classifier.hf.space/get_prediction"

    with st.spinner("Waiting for model response"):
        resp = requests.post(URL, files=files)

        if resp._content is not None:
            resp_data = json.loads(resp._content.decode('utf-8'))
            
            result = json.loads(resp_data['result'])
            label = result['label']
            confidense = result['pred_probs'][label]
            
            st.markdown(f"### Model Prediction: {class_labels[result['label']]} with {confidense*100:.1f} % confidence")