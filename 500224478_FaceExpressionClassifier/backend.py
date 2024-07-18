
from model import get_model

import torch as T
import torch.nn.functional as F
from torchvision.transforms import v2
from fastapi import FastAPI, UploadFile, File


import json
import numpy as np
from PIL import Image
from io import BytesIO

MODEL_IMAGE_WIDTH = 224
MODEL_IMAGE_HEIGHT = 224

transform = v2.Compose([
    v2.Resize((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)),
    v2.ToTensor()
])

######### Utilities #########
def load_image(image_data):
    image = Image.open(BytesIO(image_data))
    return image

def preprocess(image):
    image = image.resize((MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT))
    image = transform(image)
    
    return image

def get_prediction(image, model):
    image = T.from_numpy(np.array(image))
    print("image shape: ", image.shape)
    
    image = image.unsqueeze(0)
    # image = image.permute(0, 3, 1, 2)
    print("batch size shape: ", image.shape)

    pred_probs = model(image)
    pred_probs = F.softmax(pred_probs, dim=-1)
    pred_probs = pred_probs.detach().numpy()[0]
    label = np.argmax(pred_probs, axis=-1)

    return {
        'pred_probs': pred_probs.tolist(),
        'label': int(label)
    }

####################################

############## Backend #############
app = FastAPI()
model = T.jit.load('model_script.pt')

@app.get("/")
def foo():
    return {
        "status": "Face Expression Classifier"
    }

@app.post("/")
def bar():
    return {
        "status": "Response"
    }

@app.post("/get_prediction")
async def predict(face_img: UploadFile = File(...)):
    image = load_image(await face_img.read())

    image = preprocess(image)

    result = get_prediction(image, model)
    print("Model Predicted: \n", result)

    return {
        'result': json.dumps(result)
    }

@app.post("/test")
def test():
    return {
        'result': {
            'pred_probs': [0.5, 0.2, 0.1],
            'label': 0
        }
    }