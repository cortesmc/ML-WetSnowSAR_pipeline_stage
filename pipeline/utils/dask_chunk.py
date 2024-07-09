import numpy as np
import dask.array as da
import psutil
from dask import delayed, compute
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.util import view_as_windows
import joblib  # For loading the trained model

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
        print(windows.shape)  # Prints the shape (136, 136, 1, 15, 15, 9)

        # Change the shape to (15, 15, 9, 18496)
        windows = np.moveaxis(windows, [0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2])
        windows = windows.reshape(self.window_size, self.window_size, block.shape[2], -1)
        print(windows.shape)  # Prints the shape (15, 15, 9, 18496)

        # Reshape to match the model's expected input shape
        num_windows = windows.shape[-1]
        windows = windows.reshape(self.window_size, self.window_size, block.shape[2], num_windows)
        windows = np.moveaxis(windows, -1, 0)  # Move num_windows to the first dimension
        print(windows.shape)  # Should print (18496, 15, 15, 9)

        if self.estimator is not None:
            if self.use_predict_proba:
                predictions = self.estimator.predict_proba(windows)[:, 1]  # Assume binary classification for simplicity
            else:
                predictions = self.estimator.predict(windows)
        elif self.custom_func is not None:
            predictions = np.array([self.custom_func(window) for window in windows])
        else:
            raise ValueError("Either an estimator or a custom function must be provided")

        output_shape = (
            block.shape[0] - self.window_size + 1,
            block.shape[1] - self.window_size + 1,
        )

        output_block = predictions.reshape(output_shape)

        return output_block

    def fit(self, X, y=None):
        # No fitting process required for transformer
        return self

    def calculate_optimal_chunks(self, X):
        available_memory = psutil.virtual_memory().available
        num_cpus = psutil.cpu_count()

        element_size = X.dtype.itemsize
        element_size *= X.shape[2]  # Each element is a vector of length X.shape[2]

        # Estimate chunk size to fit in memory
        chunk_memory_size = available_memory / (num_cpus * 2)
        chunk_elements = chunk_memory_size // element_size
        chunk_side_length = int(np.sqrt(chunk_elements))

        # Ensure the chunk size is at least as large as the window size
        chunk_side_length = max(chunk_side_length, self.window_size)

        chunks = (chunk_side_length, chunk_side_length, X.shape[2])

        return chunks

    def transform(self, X):
        # Calculate optimal chunks
        chunks = self.calculate_optimal_chunks(X)

        # Use Dask to parallelize over blocks
        dask_image = da.from_array(X, chunks=chunks)
        processed_blocks = dask_image.map_blocks(
            self.transform_block, dtype=X.dtype
        )

        result = processed_blocks.compute()
        return result


if __name__ == "__main__":

    image_2d = np.random.rand(100, 100)
    # Example with sklearn estimator
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    sample_windows = view_as_windows(image_2d, (3, 3))
    # print(sample_windows.reshape(-1, 9).shape)

    X_train = sample_windows.reshape(-1, 9)
    y_train = np.random.rand(X_train.shape[0])
    model.fit(X_train, y_train)

    # Initialize the transformer for 2D image
    transformer_2d = SlidingWindowTransformer(
        estimator=model, window_size=3, padding=False
    )
    result_2d = transformer_2d.transform(image_2d)
    # print(result_2d)

    # Create a sample 3D tensor
    tensor_3d = np.random.rand(100, 100, 3)

    # Example with custom function
    def custom_mean(window):
        print(window.shape)
        return np.mean(window)

    transformer_custom = SlidingWindowTransformer(
        window_size=15, padding=False, custom_func=custom_mean
    )

    # Transform 3D tensor
    result_custom_3d = transformer_custom.transform(tensor_3d)
    # print(tensor_3d.shape)
    # print(result_custom_3d.shape)
