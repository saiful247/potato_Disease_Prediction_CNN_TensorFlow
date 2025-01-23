from fastapi import FastAPI, File, UploadFile
import uvicorn  # for running the server
import numpy as np
from io import BytesIO
from PIL import Image  # pillow
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Create an instance of FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
MODEL = tf.keras.models.load_model("../models/1.h5")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am Alive!"}

# Helper function to read and preprocess image


def read_file_as_image(data) -> np.ndarray:
    try:
        # Open the image, convert to RGB, and resize to the expected dimensions
        image = Image.open(BytesIO(data)).convert('RGB').resize((256, 256))
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Error reading the image: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        image_data = await file.read()
        image = read_file_as_image(image_data)

        # Add a batch dimension to the image array
        img_batch = np.expand_dims(image, 0)

        # Make predictions using the model
        predictions = MODEL.predict(img_batch)

        # Extract the predicted class and confidence
        index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[index]
        confidence = np.max(predictions[0])

        # Return the prediction result
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except ValueError as e:
        return {"error": str(e)}

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)  # Run the server
