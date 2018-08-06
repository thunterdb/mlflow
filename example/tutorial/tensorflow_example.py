from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mlflow
from mlflow import tensorflow, tracking
import pandas as pd
import shutil
import tempfile
import tensorflow as tf

try:
    from load_data import my_data_split
    from metrics import log_metrics
except ImportError:
    from .load_data import my_data_split
    from .metrics import log_metrics

def main(argv):
    # Builds, trains and evaluates a tf.estimator. Then, exports it for inference, logs the exported model
    # with MLflow, and loads the fitted model back as a PyFunc to make predictions.
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    (x_train, y_train, x_test, y_test) = my_data_split(seed=1)
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    # There are 13 features we are using for inference.
    feat_cols = [tf.feature_column.numeric_column(key="features", shape=(x_train.shape[1],))]
    feat_spec = {"features": tf.placeholder("float", name="features", shape=[None, x_train.shape[1]])}
    hidden_units = [50, 20]
    steps = 1000
    regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feat_cols)
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_train}, y_train, num_epochs=None, shuffle=True)
    with tracking.start_run() as tracked_run:
        mlflow.log_param("Hidden Units", hidden_units)
        mlflow.log_param("Steps", steps)
        regressor.train(train_input_fn, steps=steps)
        #test_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_test}, y_test, num_epochs=None, shuffle=True)
        # Compute mean squared error
        #mse = regressor.evaluate(test_input_fn, steps=steps)
        # Building a receiver function for exporting
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
        temp = tempfile.mkdtemp()
        try:
            saved_estimator_path = regressor.export_savedmodel(temp, receiver_fn).decode("utf-8")
            # Logging the saved model
            tensorflow.log_saved_model(saved_model_dir=saved_estimator_path, signature_def_key="predict", artifact_path="model")
            # Reloading the model
            pyfunc = tensorflow.load_pyfunc(saved_estimator_path)
            df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
            # Predicting on the loaded Python Function
            predict_df = pyfunc.predict(df)
            log_metrics(test_y=y_test, pred_y=predict_df.values)
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
