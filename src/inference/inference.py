import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd

from src.utils.utils import model_dir
from src.model.movie_predictor import MoviePredictor
from src.dataset.tmdb_dataset import TMDBRatingDataset, get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.evaluate.evaluate import evaluate


def load_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)
        
    return checkpoint


def init_model(checkpoint):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    return model, scaler


def make_inference_df(data):
    # data: [budget, popularity, runtime, vote_count]
    columns = ["budget", "popularity", "runtime", "vote_count"]
    # Add dummy target column for dataset compatibility
    data = list(data) + [0.0]
    columns.append("vote_average")
    return pd.DataFrame(
        data=[data],
        columns=columns
    )


def inference(model, scaler, data: np.array, batch_size=1):
    if data.size > 0:  # real-time inference
        df = make_inference_df(data)
        dataset = TMDBRatingDataset(df, scaler=scaler)
    else:  # batch inference
        _, _, dataset = get_datasets(scaler=scaler)

    dataloader = SimpleDataLoader(
        dataset.features, dataset.labels, batch_size=batch_size, shuffle=False
    )
    # RMSE 평가 시 더미 라벨이기 때문에 loss는 무시합니다.
    _, predictions = evaluate(model, dataloader)
    
    # 예상 평점을 소수점 2자리로 반환
    return [round(float(p), 2) for p in predictions]


def predictions_to_df(predictions):
    return pd.DataFrame(
        data=predictions,
        columns=["predicted_rating"]
    )


