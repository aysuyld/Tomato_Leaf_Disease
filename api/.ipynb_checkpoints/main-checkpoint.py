from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import os

app = FastAPI()
model = keras.models.load_model("../models/model_3")
class_names = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
               "Septoria Leaf Spot", "Spider Mites Two Spotted Spider Mite", 
               "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus",
               "Healthy"]

@app.get("/")
async def get():
    return {"message": "Welcome to the Tomato Disease Prediction API"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))
    
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    prediction = model.predict(image_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return JSONResponse(content={"class": predicted_class, "confidence": float(confidence)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
