import pickle
import numpy as np
from src.model.movie_predictor import MoviePredictor

with open(r"C:\Users\sms_1\Documents\kernelaca\p1\mlops\local_v2\models\movie_predictor\E10_T260304180143.pkl", "rb") as f:
    ck = pickle.load(f)

model = MoviePredictor(**ck["model_params"])
model.load_state_dict(ck["model_state_dict"])

scaler = ck["scaler"]

x = np.array([[50000000, 85, 120, 4000]])
x = scaler.transform(x)

pred = model.forward(x)

print(pred)
print(model.forward(scaler.transform([[10000000, 20, 90, 100]])))
print(model.forward(scaler.transform([[50000000, 80, 120, 4000]])))
print(model.forward(scaler.transform([[100000000, 95, 150, 10000]])))