import io
import os
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import MySQLdb
# from sqlalchemy.orm import sessionmaker

app = FastAPI()
classes = ['Mie', 'Oishi', 'Pop Mie']
model_path = os.path.join(os.path.dirname(__file__), 'SavedModel')

db = MySQLdb.connect(
    unix_socket='/opt/lampp/var/mysql/mysql.sock',
    user="kamu",
    password="kamu",
    database="swaloka"
)


@app.get("/all")
def get_data():
    cursor = db.cursor()
    cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id")
    result = cursor.fetchall()
    return {"data": result}


@app.get("/toko")
def get_data():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM toko")
    result = cursor.fetchall()
    return {"data": result}


@app.get("/produk")
def get_data():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM `produk`")
    result = cursor.fetchall()
    return {"data": result}


@app.get("/all/{produk}")
def get_data(produk: str):
    cursor = db.cursor()
    cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id WHERE produk.produk_name = '"+produk+"'")
    result = cursor.fetchall()
    return {"data": result}


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
        input_data = np.expand_dims(
            image_array, axis=0).astype(np.float32) / 255.0

        # Make predictions
        model_fn = model.signatures["serving_default"]
        predictions = model_fn(tf.constant(input_data))
        class_index = tf.argmax(predictions["dense"], axis=1).numpy()[0]
        class_label = classes[class_index]
        cursor = db.cursor()
        cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id WHERE produk.produk_name = '"+class_label+"'")
        result = cursor.fetchall()
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
