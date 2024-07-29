from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods =["*"],
    allow_headers=["*"]
)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, '../models/model_3')
model = keras.models.load_model(model_path)

#model = keras.models.load_model("C:\Users\aysu1\aysu\Jupyter_ile_Veri_bilimi_sonrasi_calismalar\tomato_disease\Tomato_Leaf_Disease\models\model_3")
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
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    #return JSONResponse(content={"class": predicted_class, "confidence": float(confidence)})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
