import numpy as np
import dask.array as da
import psutil
from dask import delayed, compute
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.util import view_as_windows
import joblib  

class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator=None,
        window_size=15,
        padding=True,
        use_predict_proba=False,
        custom_func=None,
    ):
        self.estimator = estimator
        self.window_size = window_size
        self.padding = padding
        self.use_predict_proba = use_predict_proba
        self.custom_func = custom_func

    def transform_block(self, block):
        if block.ndim == 3:
            pad_width = (
                (self.window_size // 2, self.window_size // 2),
                (self.window_size // 2, self.window_size // 2),
                (0, 0),
            )
        else:
            raise ValueError("Input block must be 3D")

        if self.padding:
            padded_block = np.pad(block, pad_width, mode="reflect")
        else:
            if block.shape[0] < self.window_size or block.shape[1] < self.window_size:
                return np.zeros(block.shape)
            padded_block = block

        windows = view_as_windows(
            padded_block, (self.window_size, self.window_size, block.shape[2])
        )
        windows_shape = windows.shape
        windows = windows.reshape(-1, self.window_size, self.window_size, block.shape[2])
        if self.estimator is not None:
            if self.use_predict_proba:
                predictions = self.estimator.predict_proba(windows)[:,1]
                print(predictions.shape)
            else:
                predictions = self.estimator.predict(windows)
        elif self.custom_func is not None:
            predictions = np.array([self.custom_func(window) for window in windows])
        else:
            raise ValueError("Either an estimator or a custom function must be provided")
        predictions = predictions.reshape((windows_shape[0],windows_shape[1],1))
        return predictions

    def fit(self, X, y=None):
        # No fitting process required for transformer
        return self

    def calculate_optimal_chunks(self, X):
        available_memory = psutil.virtual_memory().available
        num_cpus = psutil.cpu_count()

        element_size = X.dtype.itemsize
        element_size = element_size* 15*15*9 

        chunk_memory_size = available_memory / (num_cpus*2)
        chunk_elements = chunk_memory_size // element_size
        chunk_side_length = int(np.sqrt(chunk_elements))

        chunk_side_length = max(chunk_side_length, self.window_size)
        chunk_side_length = int(chunk_side_length/self.window_size)*self.window_size

        chunks = (chunk_side_length, chunk_side_length, X.shape[2])

        return chunks

    def transform(self, X):
        chunks = self.calculate_optimal_chunks(X)
        chunks = (225,225,9) # TODO 
        print(f"chunks : {chunks}")
        dask_image = da.from_array(X, chunks=chunks)
        processed_blocks = dask_image.map_blocks(
            self.transform_block,  dtype=np.float64, enforce_ndim=False
        )

        result = processed_blocks.compute()
        return result