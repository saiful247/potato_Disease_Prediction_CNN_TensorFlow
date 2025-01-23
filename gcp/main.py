from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

# Here you need to put the name of your GCP bucket
BUCKET_NAME = "ml-model-tf-saif"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict_disease(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/1.h5",
            "/tmp/1.h5",
        )
        model = tf.keras.models.load_model("/tmp/1.h5")

    image = request.files["file"]

    # Preprocess the image
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))  # image resizing
    )

    image = image / 255.0  # normalize the image in 0 to 1 range

    # Add batch dimension
    img_array = tf.expand_dims(image, 0)

    # Make predictions
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    # Convert to native float
    confidence = round(100 * float(np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
