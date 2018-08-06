"""
This contains all the validation metrics for this experiment.

It is important that they all be agreed upon, hence shared into
a single file.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def log_metrics(test_y, pred_y):
    (rmse, mae, r2) = eval_metrics(test_y, pred_y)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

