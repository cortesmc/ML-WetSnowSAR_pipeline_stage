from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BandSelector(BaseEstimator, TransformerMixin):
    def __init__(self, bands=[0]):
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X[:, :, :, self.bands].shape)
        return X[:, :, :, self.bands]
