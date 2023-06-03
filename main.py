import io
import os
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()
classes = ['Mie Goreng', 'Oishi', 'Pop Mie']
model_path = os.path.join(os.path.dirname(__file__), 'SavedModel')

@app.post("/make-predictions")
async def make_predictions(file: UploadFile = File(...)):
    try:
        # Load model
        model = tf.saved_model.load(model_path)

        # Read and preprocess the uploaded image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        image = image.resize((224, 224))
        image_array = np.array(image)
        input_data = np.expand_dims(image_array, axis=0).astype(np.float32) / 255.0

        # Make predictions
        model_fn = model.signatures["serving_default"]
        predictions = model_fn(tf.constant(input_data))
        class_index = tf.argmax(predictions["dense"], axis=1).numpy()[0]
        class_label = classes[class_index]

        return {"prediction": class_label}

    except Exception as e:
        return {"error": str(e)}
