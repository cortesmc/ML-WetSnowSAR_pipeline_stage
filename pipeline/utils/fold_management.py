import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import itertools
import random

def simple_split(x, train_size=0.8, rng=None, shuffle=False):
    """
    Perform a simple split of the data into training and testing sets.

    Parameters:
    - x : array-like
        The data to split.
    - train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the training split.
    - rng : int or None, optional (default=None)
        rng used by the random number generator for reproducibility.
    - shuffle : bool, optional (default=False)
        Whether to shuffle the data before splitting.

    Returns:
    - tuple
        A tuple containing train and test indices.
    """
    n_samples = len(x)
    if shuffle:
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_size = int(train_size * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return [(train_indices, test_indices)]

def KFold_method(x, train_size=0.8, rng=None, shuffle=False):
    """
    Perform K-Fold cross-validation on the data.
    
    Parameters:
    - x : array-like
        The data to split into K folds.
    - train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    - rng : int or None, optional (default=None)
        rng used by the random number generator for reproducibility. If None, a random rng will be selected.
    - shuffle : bool, optional (default=False)
        Whether to shuffle the data before splitting into folds.

    Returns:
    - list of tuples
        A list containing train and test indices for each fold.
    """
    if shuffle == False:
        rng = None

    n_splits = int(1/(1-train_size))
    kf = KFold(n_splits=n_splits, random_state=rng, shuffle=shuffle)
    
    return list(kf.split(x))

def fold_massive_method(dict_massives):
    """
    Generate train and test indices for each massive in a dictionary.

    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.

    Returns:
    - list of tuples
        A list containing train and test indices for each massive.
    """
    unique_massives = list(dict_massives.keys())
    result = []

    for i in range(len(unique_massives)):
        test_massive = unique_massives[i]
        train_indices = []
        test_indices = dict_massives[test_massive]['indices']
        for j in range(len(unique_massives)):
            if j != i:
                train_indices.extend(dict_massives[unique_massives[j]]['indices'])
        
        result.append((train_indices, test_indices))
    
    return result

def combination_method(dict_massives, train_size=0.8, proximity_value=1, shuffle=False, rng=None):
    """
    Generate prioritized combinations of massives based on a given dictionary.

    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.
        Example: {
            'massif1': {'count': 10, 'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            'massif2': {'count': 5, 'indices': [10, 11, 12, 13, 14]},
            ...
        }
    - train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    - proximity_value : int, optional (default=1)
        A value to control the proximity to the desired train size.
    - shuffle : bool, optional (default=False)
        Whether to shuffle the selection of combinations.
    - rng : int, optional (default=None)
        rng for random number generator (used if shuffle is True).

    Returns:
    - list of tuples
        A list containing train and test indices for each prioritized combination of massives.
    """
    if rng is None:
        rng = np.random.RandomState()

    total_count = sum(value['count'] for value in dict_massives.values())
    massives = list(dict_massives.keys())
    massives.sort()

    all_combinations = []
    for r in range(1, len(massives) + 1):
        combinations_object = itertools.combinations(massives, r)
        combinations_list = list(combinations_object)
        all_combinations.extend(combinations_list)

    valid_combinations = []
    for combo in all_combinations:
        combo_count = sum(dict_massives[massif]['count'] for massif in combo)
        percentage = (combo_count / total_count) * 100
        if (train_size * 100) - proximity_value <= percentage <= (train_size * 100) + proximity_value:
            valid_combinations.append(combo)
    
    valid_combinations.sort()

    if shuffle:
        rng.shuffle(valid_combinations)

    uncovered_train_massives = set(massives)
    uncovered_test_massives = set(massives)
    selected_combinations = []

    for combo in valid_combinations:
        if not uncovered_train_massives and not uncovered_test_massives:
            break

        train_massifs_in_combo = set(combo)
        test_massifs_in_combo = set(massives) - train_massifs_in_combo

        if uncovered_train_massives & train_massifs_in_combo or uncovered_test_massives & test_massifs_in_combo:
            selected_combinations.append(combo)
            uncovered_train_massives -= train_massifs_in_combo
            uncovered_test_massives -= test_massifs_in_combo

    result = []
    for combo in selected_combinations:
        train_indices = []
        test_indices = []
        for massif in massives:
            if massif in combo:
                train_indices.extend(dict_massives[massif]['indices'])
            else:
                test_indices.extend(dict_massives[massif]['indices'])

        result.append((train_indices, test_indices))

    return result

class FoldManagement: 
    """
    A class to manage the creation of the folds for trainning.

    Attributes:
    - method : str
        The fold method to use.
    - shuffle : bool
        Gives the user the choice to shuffle the data output and the fold creation.
    - rng : int
        rng used for the random creation.
    - train_aprox_size : float
        The balance of that between trinning and test datasets.

    Methods:
    - Split(x, y)
        Apply the selected labeling method to the provided metadata.
    """

    def __init__(self, method="kFold", shuffle=False, rng=None, train_aprox_size=0.8):
        self.method = method
        self.shuffle = shuffle
        self.rng = rng
        self.train_aprox_size = train_aprox_size
        self.massives_count = {}
        self.results = None

    def split(self, x, y):
        """
        Split the data into train and test sets based on the specified folding method.

        .. warning::
        The method combinationFold takes the variable: "proximity_value"; which is a value to control the proximity of the distribution of the folds.
        
        Parameters:
        - x : array-like
            The data to split.
        - y : dict
            The metadata associated with the data.

        Returns:
        - list of tuples
            Generator like list.
        """
        for index, name in enumerate(y['metadata'][:, 1]):
            if name not in self.massives_count:
                self.massives_count[name] = {'count': 0, 'indices': []}
            self.massives_count[name]['count'] += 1
            self.massives_count[name]['indices'].append(index)
            
        if ((np.unique(y['metadata'][:, 1]).size == 1) and (self.method != "kFold")):
            self.method = "kFold"
            return self.split(x=x, y=y)
        
        match self.method: 
            case "kFold":
                self.results = KFold_method(x, train_size=self.train_aprox_size, rng=self.rng, shuffle=self.shuffle)

            case "mFold":
                self.results = fold_massive_method(self.massives_count)

            case "combinationFold":
                self.results = combination_method(self.massives_count, train_size=self.train_aprox_size, proximity_value=1, shuffle=self.shuffle, rng=self.rng)

            case "simpleSplit":
                self.results = simple_split(x, train_size=self.train_aprox_size, rng=self.rng, shuffle=self.shuffle)

        return self.results
