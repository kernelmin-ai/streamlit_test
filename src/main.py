import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import wandb  #
from dotenv import load_dotenv  #
import fire
from alive_progress import alive_it
from icecream import ic
from tqdm import tqdm
import numpy as np

from src.dataset.data_loader import SimpleDataLoader
from src.dataset.watch_log import get_datasets
from src.evaluate.evaluate import evaluate
from src.model.movie_predictor import MoviePredictor, model_save
from src.train.train import train
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.utils.factory import ModelFactory
from src.inference.inference import (
    load_checkpoint,
    init_model,
    inference,
    predictions_to_df
)
from src.postprocess.postprocess import write_db

init_seed()
load_dotenv()  #

def get_latest_run(project_name):
    try:
        runs = wandb.Api().runs(path=project_name, order="-created_at")
        if not runs:
            return f"{project_name}-000"
        return runs[0].name
    except Exception:
        return f"{project_name}-000"


def run_train(model_name, num_epochs=10, batch_size=64, hidden_dim=64):
    """
    this is run_train definition.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=api_key)

    project_name = "teamfirst-rating-prediction"
    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="TMDB 평점 예측 베이스라인 실험",
        tags=["regression", "rating", "predict"],
        config=locals(),
    )

    # 데이터셋 및 DataLoader 생성 (TMDB 데이터셋으로 교체)
    from src.dataset.tmdb_dataset import get_datasets, TMDBRatingDataset
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(
        train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True
    )
    val_loader = SimpleDataLoader(
        val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=False
    )
    test_loader = SimpleDataLoader(
        test_dataset.features, test_dataset.labels, batch_size=batch_size, shuffle=False
    )

    # 모델 초기화 (Regression 구조로 변경)
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": 1, # 회귀는 출력 차원이 1입니다.
        "hidden_dim": hidden_dim,
    }
    # model = MoviePredictor(**model_params)
    model = ModelFactory.create(model_name, **model_params)

    # 학습 루프
    for epoch in alive_it(range(num_epochs)):
        time.sleep(0.2)
        train_rmse = train(model, train_loader)
        val_rmse, _ = evaluate(model, val_loader)

        wandb.log({"RMSE/Train": train_rmse})
        wandb.log({"RMSE/Valid": val_rmse})

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train RMSE: {train_rmse:.4f}, "
            f"Val RMSE: {val_rmse:.4f}, "
            f"Val-Train RMSE : {val_rmse-train_rmse:.4f}"
        )

    # 테스트
    test_rmse, predictions = evaluate(model, test_loader)
    print(f"Final Test RMSE: {test_rmse:.4f}")
    # ic(test_loss)
    # ic([train_dataset.decode_content_id(idx) for idx in predictions])

    saved_model_path = model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        loss=test_rmse,
        scaler=train_dataset.scaler,
        label_encoder=None,
    )

    # WandB Artifact 생성 및 추가
    artifact = wandb.Artifact(
        name=f"{project_name}-model",
        type="model",
        description="Content-based movie prediction model"
    )
    artifact.add_file(saved_model_path)
    
    # Artifact 로깅
    wandb.log_artifact(artifact, aliases=["latest", f"epoch_{num_epochs}"])

    wandb.finish()


def run_inference(data=None, batch_size=64):
    checkpoint = load_checkpoint()
    model, scaler = init_model(checkpoint)

    if data is None:
        data = []

    data = np.array(data)

    predictions = inference(model, scaler, data, batch_size)
    print(predictions)

    write_db(predictions_to_df(predictions), "mlops", "rating_predictions")


if __name__ == "__main__":
    fire.Fire({
        "train": run_train,  # python main.py train --model_name movie_predictor
        "inference": run_inference,  # python main.py inference -b 32
    })

