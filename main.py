import io
import os
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import MySQLdb
import methode
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



from fastapi import HTTPException

@app.get("/all")
def get_all_data():
    mydict = methode.allin()
    cursor = db.cursor()
    cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id")
    result = cursor.fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")

    mydict.add(result)
    return mydict.data


@app.get("/toko")
def get_toko_data():
    mydict = methode.toko()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM toko")
    result = cursor.fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Data toko tidak ditemukan")

    mydict.add(result)
    return mydict.data


@app.get("/produk")
def get_produk_data():
    mydict = methode.product()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM `produk`")
    result = cursor.fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Data produk tidak ditemukan")

    mydict.add(result)
    return mydict.data


@app.get("/all/{produk}")
def get_specific_data(produk: str):
    mydict = methode.allin()
    cursor = db.cursor()
    cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id WHERE produk.produk_name = '"+produk+"'")
    result = cursor.fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Produk tidak ditemukan")

    mydict.add(result)
    return mydict.data



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
        mydict = methode.allin()
        cursor = db.cursor()
        cursor.execute("SELECT produk_toko.id,produk.produk_name,produk.price,toko.name,toko.longtitude,toko.latitiude FROM produk_toko INNER JOIN toko ON produk_toko.toko_id = toko.id INNER JOIN produk ON produk_toko.produk_id = produk.id WHERE produk.produk_name = '"+class_label+"'")
        result = cursor.fetchall()
        mydict.add(result)
        return mydict.data

    except Exception as e:
        return {"error": str(e)}
