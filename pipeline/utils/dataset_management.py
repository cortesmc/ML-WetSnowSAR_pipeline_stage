import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
import os, ast

from utils.files_management import load_h5


def random_shuffle(X, y, rng=-1):
    """Shuffle randomly the dataset

    Parameters
    ----------
    X : numpy array
        dataset of images

    y : numpy array
        dataset of labels

    rng : int, optional
        Random seed, by default -1, must be a np.random.default_rng() object

    Returns
    -------
    numpy array
        shuffled dataset of images

    numpy array
        shuffled dataset of labels

    """
    if rng == -1:
        rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

def balance_dataset(X, Y, shuffle=False):
    """Balance the dataset by taking the minimum number of samples per class (under-sampling)

    Parameters
    ----------
    X : numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands)

    Y : numpy array
        dataset of labels in string, shape (n_samples,)

    shuffle : bool, optional
        Shuffle the dataset, by default False

    Returns
    -------
    numpy array
        balanced dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        balanced dataset of labels in string, shape (n_samples,)
    """
    if shuffle:
        X, Y = random_shuffle(X, Y)
    cat, counts = np.unique(Y, return_counts=True)
    min_count = np.min(counts)
    X_bal = []
    Y_bal = []
    for category in cat:
        idx = np.where(Y == category)[0]
        idx = idx[:min_count]
        X_bal.append(X[idx])
        Y_bal.append(Y[idx])
    X_bal = np.concatenate(X_bal)
    Y_bal = np.concatenate(Y_bal)
    return X_bal, Y_bal

def balance_dataset_with_imblearn(X, Y, method='under', shuffle=False):
    """Balance the dataset using specified method (under-sampling, over-sampling, or combined)

    Parameters
    ----------
    X : numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands)

    Y : numpy array
        dataset of labels in string, shape (n_samples,)

    method : str, optional
        Method to balance the dataset, 'under' for under-sampling, 'over' for over-sampling,
        and 'combine' for a combination of over- and under-sampling, by default 'under'

    shuffle : bool, optional
        Shuffle the dataset, by default False

    Returns
    -------
    numpy array
        balanced dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        balanced dataset of labels in string, shape (n_samples,)
    """
    if shuffle:
        X, Y = random_shuffle(X, Y)
    
    if method == 'under':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'over':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'combine':
        sampler = SMOTEENN(random_state=42)
    else:
        raise ValueError("Method should be 'under', 'over', or 'combine'")

    X_res, Y_res = sampler.fit_resample(X.reshape(X.shape[0], -1), Y)
    X_res = X_res.reshape(-1, X.shape[1], X.shape[2], X.shape[3])

    return X_res, Y_res

class BFold:
    """Balanced Fold cross-validator, only valid for binary classification.

    Split dataset into k consecutive folds (without shuffling by default),
    ensuring that each fold has the same number of samples from each class.
    It allows to have B sub-datasets balanced, and have a complete view of the
    data.

    Parameters
    ----------
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Attributes
    ----------
    n_splits : int
        Returns the number of splitting iterations in the cross-validator.

    """

    def __init__(self, shuffle=False, random_state=42):
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)

    def get_n_splits(self, X, y, groups=None):
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        self.n_splits = int(majority_numb / minority_numb)
        return self.n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training set.

        Parameters
        ----------
        X : numpy array
            dataset of images

        y : numpy array
            dataset of labels        test : numpy array
            The testing set indices for that split.
        ------
        train : numpy array
            The training set indices for that split.
        """
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        minority_class = np.unique(y, return_counts=True)[0][argmin_minor]
        ratio = majority_numb / minority_numb
        if majority_numb % minority_numb == 0:
            self.n_splits = int(ratio)
        else:
            self.n_splits = int(ratio) + 1

        idx_minority = np.where(y == minority_class)[0]
        idx_majority = np.where(y != minority_class)[0]

        if self.shuffle:
            self.rng.shuffle(idx_majority)

        for i in range(self.n_splits):
            start = i * minority_numb
            end = (i + 1) * minority_numb
            if i == self.n_splits - 1:
                end = len(idx_majority)
                miss = minority_numb - (end - start)
                idx_majority_balanced = np.concatenate(
                    [idx_majority[start:end], idx_majority[:miss]]
                )
            else:
                idx_majority_balanced = idx_majority[start:end]

            idx_train = np.concatenate([idx_minority, idx_majority_balanced])
            self.rng.shuffle(idx_train)
            yield idx_train


def parse_pipeline(args, idx):
    """Parse a dictionary to create a pipeline of estimators
    The dictionary must have the following structure::
    {
        "import": [
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.decomposition import PCA",
            "from sklearn.svm import SVC",
        ],
        "pipeline": [
            [
                ["StandardScaler", {"with_mean": False, "with_std": False}],
                ["PCA", {"n_components": 0.95}],
                ["SVC", {"kernel": "rbf", "C": 10, "gamma": 0.01}],
            ]
        ],
    }

    Parameters
    ----------
    args : dict
        Dictionary containing the pipeline

    idx : int
        Index of the pipeline to use in case of multiple pipelines
        analysis

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline of estimators
    """
    for import_lib in args.import_list :
        exec(import_lib)
    pipe = ast.literal_eval(args.pipeline[idx])
    print(pipe)
    step = []
    for i in range(1,len(pipe)):
        name_methode = pipe[i][0]
        estim = locals()[name_methode]()

        if len(pipe[i]) > 1:
            [
                [
                    setattr(estim, param, pipe[i][g][param])
                    for param in pipe[i][g].keys()
                ]
                for g in range(1, len(pipe[i]))
            ]
        step.append((name_methode, estim))
    return Pipeline(step, verbose=True, memory=".cache")


def load_train(i_path, bands_max, balanced, shffle=True, encode=True):
    """Load a hdf5 file containing the training dataset

    Parameters
    ----------
    i_path : str
        Path to the hdf5 file with name "data_train.h5"

    bands_max : list
        List of bands to keep in the dataset

    balanced : bool
        If True, the dataset is balanced

    shffle : bool, optional
        If True, the dataset is shuffled (rng 42), by default True

    encode : bool, optional
        If True, the labels are encoded, by default True

    Returns
    -------
    numpy array
        Dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        Dataset of labels in string, shape (n_samples,)

    sklearn.preprocessing.LabelEncoder
        Encoder used to encode the labels
    """

    X_train, Y_train = load_h5(os.path.join(i_path, "data_train.h5"))
    if bands_max != -1:
        X_train = X_train[:, :, :, bands_max]
    if balanced:
        X_train, Y_train = balance_dataset(X_train, Y_train, shuffle=shffle)
    if encode:
        encoder = LabelEncoder()
        encoder.fit(Y_train)
        Y_train = encoder.transform(Y_train)
    else:
        encoder = None
    return X_train, Y_train, encoder


def load_test(i_path, bands_max, balanced, shffle=True, encoder=None):
    """Load a hdf5 file containing the testing dataset

    Parameters
    ----------
    i_path : str
        Path to the hdf5 file with name "data_test.h5"

    bands_max : list
        List of bands to keep in the dataset

    balanced : bool
        If True, the dataset is balanced

    shffle : bool, optional
        If True, the dataset is shuffled (rng 42), by default True

    encoder : sklearn.preprocessing.LabelEncoder, optional
        Encoder used to encode the labels, by default None
        The encoder must be fitted before ie using the `load_train` function

    Returns
    -------
    numpy array
        Dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        Dataset of labels in string, shape (n_samples,)
    """

    X_test, Y_test = load_h5(os.path.join(i_path, "data_test.h5"))
    X_test = X_test[:, :, :, bands_max]
    if balanced:
        X_test, Y_test = balance_dataset(X_test, Y_test, shuffle=shffle)
    if encoder is not None:
        Y_test = encoder.transform(Y_test)
    return X_test, Y_test
