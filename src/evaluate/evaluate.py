import numpy as np


def evaluate(model, val_loader):
    total_squared_error = 0
    total_samples = 0
    all_predictions = []

    for features, labels in val_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)

        batch_squared_error = np.sum((predictions - labels) ** 2)
        total_squared_error += batch_squared_error
        total_samples += len(features)
        
        # Regression does not use argmax, just store the predicted float values
        all_predictions.extend(predictions.flatten())

    mse = total_squared_error / total_samples
    rmse = np.sqrt(mse)
    return rmse, all_predictions
