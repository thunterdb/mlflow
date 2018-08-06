import warnings
import sys

from sklearn.linear_model import ElasticNet

try:
    from load_data import my_data_split
    from metrics import log_metrics
except ImportError:
    from .load_data import my_data_split
    from .metrics import log_metrics

import mlflow.sklearn
import mlflow


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    (train_x, train_y, test_x, test_y) = my_data_split(seed=1)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)


        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.sklearn.log_model(lr, "model")
        log_metrics(test_y, predicted_qualities)
