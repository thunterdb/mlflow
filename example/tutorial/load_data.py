"""
Loads the data into pandas dataframe.

We use a built-in dataset (Boston housing).
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd


def my_data():
    boston = load_boston()
    bos = pd.DataFrame(boston.data)
    bos.columns = boston.feature_names
    bos['PRICE'] = boston.target
    return bos

def my_data_split(seed=1):
    data = my_data()
    train, test = train_test_split(data, random_state=seed)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["PRICE"], axis=1)
    test_x = test.drop(["PRICE"], axis=1)
    train_y = train[["PRICE"]]
    test_y = test[["PRICE"]]
    return train_x, train_y, test_x, test_y
