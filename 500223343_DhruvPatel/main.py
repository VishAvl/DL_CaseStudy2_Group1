from fastapi import FastAPI, UploadFile, File
import json
from PIL import Image
from io import BytesIO
import numpy as np
from model import build_model
app = FastAPI()

#Load model
image_shape = (224,224,3)
num_classes = 6
model = build_model(image_shape, num_classes)
model.load_weights('./new_model_weights.h5')
classes = {
    0: 'Ahegao',
    1: 'Angry',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad',
    5: 'Surprise'
}

@app.get("/")
def first_api():
    return {
        "response": "Face Expression Prediction"
    }

@app.post("/prediction")
async def prediction(image: UploadFile = File(...)):
    image = await image.read()

    # process image
    image = Image.open(BytesIO(image))
    image = image.resize((image_shape[0], image_shape[1]))


    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    label = np.argmax(prediction, axis=-1).tolist()

    return {
        "label": label,
        "class": classes[label]
    }