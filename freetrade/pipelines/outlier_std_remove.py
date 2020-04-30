import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierStdRemove(BaseEstimator, TransformerMixin):

    def __init__(self, std_threshold):
        self.std_threshold = std_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[X.iloc[:, 1:6].apply(lambda x: np.abs(x - x.mean()) / x.std() < self.std_threshold).all(axis=1)]
        return X
