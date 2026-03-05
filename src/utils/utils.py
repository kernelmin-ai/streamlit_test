import os
import random

import numpy as np


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():  # pd.read_csv("./dataset/watch_log.csv")
    return os.path.join(  # /opt/mlops/src/utils/../..  -> /opt/mlops
        os.path.dirname(  # /opt/mlops/src/utils
            os.path.abspath(__file__)  # /opt/mlops/src/utils/utils.py
        ),
        "..",
        ".."
    )


def model_dir(model_name):
    return os.path.join(  # /opt/mlops/models/{model_name}
        project_path(),
        "models",
        model_name
    )

def auto_increment_run_suffix(name: str, pad=3):  # movie-predictor-001, movie-predictor-002, ...
    suffix = name.split("-")[-1]  # "001": str
    next_suffix = str(int(suffix) + 1).zfill(pad)  # "001" -> 1 + 1 -> 2 -> "2" -> "002"
    return name.replace(suffix, next_suffix)
