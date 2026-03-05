import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_DIR)

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.inference.inference import load_checkpoint, init_model, inference
from src.postprocess.postprocess import read_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

load_dotenv()
checkpoint = load_checkpoint()
model, scaler, label_encoder = init_model(checkpoint)

class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

@app.post("/predict")  # real-time inference endpoint
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity,
        ])

        recommend = inference(
            model=model, 
            scaler=scaler, 
            label_encoder=label_encoder, 
            data=data
        )
        recommend = [int(r) for r in recommend]
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # 2xx: ok, 4xx: bad request, 5xx: internal server error


@app.get("/batch-predict")  # batch inference endpoint
async def batch_predict(k: int = 5):  # 43.128.7.34:8080/batch-predict?k=10&m=10
    try:
        recommend = read_db("mlops", "recommend", k=k)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WSGI, ASGI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # http://www.naver.com => DNS => 43.128.42.38:80
