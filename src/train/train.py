import numpy as np


def train(model, train_loader):
    total_squared_error = 0
    total_samples = 0
    for features, labels in train_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)
        
        # 정밀한 RMSE 계산을 위해 배치별 제곱 오차 합계를 누적합니다.
        batch_squared_error = np.sum((predictions - labels) ** 2)
        total_squared_error += batch_squared_error
        total_samples += len(features)
        
        model.backward(features, labels, predictions)

    mse = total_squared_error / total_samples
    rmse = np.sqrt(mse)
    return rmse
