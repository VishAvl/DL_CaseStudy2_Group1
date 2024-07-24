import streamlit as st
import requests
import json
from io import BytesIO

st.title('Face Expression Classifier')

image_file = st.file_uploader("Upload Image")

if image_file is not None:

    image = image_file.getvalue()

    response = requests.post(
        "https://shiv8587-face-expression.hf.space",
        files = {
            "image": BytesIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Patient is {label['class']}")