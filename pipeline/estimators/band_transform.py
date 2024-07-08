from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BandSelector(BaseEstimator, TransformerMixin):
    def __init__(self, bands=[0]):
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :, self.bands]

class BandTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bands=[], transformations=None):
        self.bands = bands
        self.transformations = transformations if transformations is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = np.copy(X)
        
        bands_to_transform = self.bands if self.bands else range(X.shape[3])
        
        for band in bands_to_transform:
            for transformation in self.transformations:
                match transformation:
                    case 'log':
                        X_transformed[:, :, band]  = np.log(X_transformed[:, :, band] )
                    case 'exp':
                        X_transformed[:, :, band]  = np.exp(X_transformed[:, :, band] )
                    case 'ln':
                        X_transformed[:, :, band]  = np.log1p(X_transformed[:, :, band] )
                    case _:
                        raise ValueError(f"Transformation '{transformation}' is not supported.")
        
        return X_transformed