import numpy as np
import dask.array as da
import psutil
from dask import delayed, compute
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.util import view_as_windows


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator=None,
        window_size=3,
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
        if block.ndim == 2:
            pad_width = (
                (self.window_size // 2, self.window_size // 2),
                (self.window_size // 2, self.window_size // 2),
            )
        elif block.ndim == 3:
            pad_width = (
                (self.window_size // 2, self.window_size // 2),
                (self.window_size // 2, self.window_size // 2),
                (0, 0),
            )
        else:
            raise ValueError("Input block must be either 2D or 3D")

        if self.padding:
            padded_block = np.pad(block, pad_width, mode="reflect")
        else:
            if block.shape[0] < self.window_size or block.shape[1] < self.window_size:
                return np.zeros(block.shape)  # Handle edge case
            padded_block = block

        if block.ndim == 2:
            windows = view_as_windows(
                padded_block, (self.window_size, self.window_size)
            )
            windows = windows.reshape(-1, self.window_size * self.window_size)
        else:  # 3D case
            windows = view_as_windows(
                padded_block, (self.window_size, self.window_size, block.shape[2])
            )
            windows = windows.reshape(
                -1, self.window_size * self.window_size * block.shape[2]
            )

        if self.estimator is not None:
            if self.use_predict_proba:
                predictions = self.estimator.predict_proba(windows)[
                    :, 1
                ]  # Assume binary classification for simplicity
            else:
                predictions = self.estimator.predict(windows)
        elif self.custom_func is not None:
            predictions = np.array([self.custom_func(window) for window in windows])
        else:
            raise ValueError(
                "Either an estimator or a custom function must be provided"
            )

        if self.padding:
            output_shape = block.shape
        else:
            if block.ndim == 2:
                output_shape = (
                    block.shape[0] - self.window_size + 1,
                    block.shape[1] - self.window_size + 1,
                )
            else:
                output_shape = (
                    block.shape[0] - self.window_size + 1,
                    block.shape[1] - self.window_size + 1,
                    block.shape[2],
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
        if X.ndim == 2:
            element_size *= 1  # Each element is a single value
        elif X.ndim == 3:
            element_size *= X.shape[2]  # Each element is a vector of length X.shape[2]

        # Estimate chunk size to fit in memory
        chunk_memory_size = available_memory / (num_cpus * 2)
        chunk_elements = chunk_memory_size // element_size
        chunk_side_length = int(np.sqrt(chunk_elements))

        # Ensure the chunk size is at least as large as the window size
        chunk_side_length = max(chunk_side_length, self.window_size)

        if X.ndim == 2:
            chunks = (chunk_side_length, chunk_side_length)
        else:  # 3D case
            chunks = (chunk_side_length, chunk_side_length, X.shape[2])

        return chunks

    def transform(self, X):
        # Calculate optimal chunks
        chunks = self.calculate_optimal_chunks(X)

        # Use Dask to parallelize over blocks
        dask_image = da.from_array(X, chunks=chunks)
        if X.ndim == 3:
            processed_blocks = dask_image.map_blocks(
                self.transform_block, drop_axis=[-1]
            )

        else:
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
    X_train = sample_windows.reshape(-1, 9)
    y_train = np.random.rand(X_train.shape[0])
    model.fit(X_train, y_train)

    # Initialize the transformer for 2D image
    transformer_2d = SlidingWindowTransformer(
        estimator=model, window_size=3, padding=False
    )
    result_2d = transformer_2d.transform(image_2d)
    print(result_2d)

    # Create a sample 3D tensor
    tensor_3d = np.random.rand(100, 100, 3)

    # Example with custom function
    def custom_mean(window):

        return np.mean(window, keepdims=True)[:, np.newaxis]

    transformer_custom = SlidingWindowTransformer(
        window_size=3, padding=False, custom_func=custom_mean
    )

    # Transform 3D tensor
    result_custom_3d = transformer_custom.transform(tensor_3d)
    print(result_custom_3d)