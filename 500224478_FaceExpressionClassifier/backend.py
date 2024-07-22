import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, UploadFile, File
import json
from PIL import Image
from io import BytesIO
import numpy as np

from model import get_model

app = FastAPI()

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

MODEL_WEIGHT_PATH = 'vgg_face_weights2.h5'
model = get_model(
		image_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3),
		num_classes = 6,
		model_weights = MODEL_WEIGHT_PATH
	)
print(model.summary())
print("Model Loaded Successfully")

######### Utilities #########
def load_image(image_data):
	image = Image.open(BytesIO(image_data))
	return image

def preprocess(image):
	image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

	image = np.array(image)
	image = np.expand_dims(image, axis=0) / 255.

	return image

def get_prediction(image):
	probs = model.predict(image)[0]
	label = np.argmax(probs)

	return {
		'pred_probs': probs.tolist(),
		'label': int(label)
	}

@app.get("/")
def foo():
	return {
		"status": "Face Expression Classifier"
	}


@app.post("/get_prediction")
async def predict(face_img: UploadFile = File(...)):
	image = load_image(await face_img.read())

	image = preprocess(image)
	result = get_prediction(image)

	return {
		"result": json.dumps(result)
	}