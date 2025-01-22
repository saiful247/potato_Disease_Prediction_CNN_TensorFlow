from fastapi import FastAPI, File, UploadFile

import uvicorn  # for running the server
import numpy as np
from io import BytesIO

from PIL import Image  # pillow

import tensorflow as tf

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()  # Create an instance of FastAPI


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("../models/1.h5")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get("/ping")
async def ping():
    return "Hello, i am Alive!"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # async , wait used for when many users trying to send something reduce the problem of traffic - study this
    image = read_file_as_image(await file.read())

    # image is not a single 1 it is a batch image
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    # predictions[0]  0 mean first image. usually in this batch only 1 image available
    index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class, 'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)  # Run the server
