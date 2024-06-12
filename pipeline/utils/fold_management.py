import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import itertools

def KFold_method(x, train_size=0.8, seed=None, shuffle=False):
    """
    Perform K-Fold cross-validation on the data.
    
    Parameters:
    - x : array-like
        The data to split into K folds.
    - train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    - seed : int or None, optional (default=None)
        Seed used by the random number generator for reproducibility. If None, a random seed will be selected.
    - shuffle : bool, optional (default=False)
        Whether to shuffle the data before splitting into folds.

    Returns:
    - list of tuples
        A list containing train and test indices for each fold.
    """
    if shuffle == False:
        seed = None

    n_splits = int(1/(1-train_size))
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=shuffle)
    
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

def combination_method(dict_massives, train_size=0.8, proximity_value=1):
    """
    Generate prioritized combinations of massives based on a given dictionary.

    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.
    - train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    - proximity_value : int, optional (default=1)
        A value to control the proximity to the desired train size.

    Returns:
    - list of tuples
        A list containing train and test indices for each prioritized combination of massives.
    """
    total_count = sum(value['count'] for value in dict_massives.values())
    massives = list(dict_massives.keys())

    all_combinations = []
    for r in range(1, len(massives)):
        combinations_object = itertools.combinations(massives, r)
        combinations_list = list(combinations_object)
        all_combinations.extend(combinations_list)

    valid_combinations = []
    for combo in all_combinations:
        combo_count = sum(dict_massives[massif]['count'] for massif in combo)
        percentage = (combo_count / total_count) * 100
        if (train_size * 100) - proximity_value <= percentage <= (train_size * 100) + proximity_value:
            valid_combinations.append(combo)

    valid_combinations.sort(key=lambda combo: len(massives) - len(combo))

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

def balance_classes(results, targets, method='oversample',seed = 42):
    """
    Balance the classes within each fold using the specified method.

    Parameters
    ----------
    results : list of tuples
        A list containing train and test indices for each fold.
    targets : numpy.ndarray
        Target labels.
    method : str, optional (default='oversample')
        The resampling method to use ('oversample', 'undersample', 'smote').

    Returns
    -------
    list of tuples
        A list containing balanced train and test indices for each fold.
    """
    balanced_results = []

    for train_indices, test_indices in results:
        train_targets = targets[train_indices]
        
        if method == 'oversample':
            unique_classes, class_counts = np.unique(train_targets, return_counts=True)
            max_class_count = class_counts.max()

            balanced_train_indices = []

            for cls in unique_classes:
                cls_indices = np.array(train_indices)[train_targets == cls]
                if len(cls_indices) < max_class_count:
                    cls_indices = resample(cls_indices, replace=True, n_samples=max_class_count, random_state=seed)
                balanced_train_indices.extend(cls_indices)

            balanced_train_indices = np.array(balanced_train_indices)
        
        elif method == 'undersample':
            unique_classes, class_counts = np.unique(train_targets, return_counts=True)
            min_class_count = class_counts.min()

            balanced_train_indices = []

            for cls in unique_classes:
                cls_indices = np.array(train_indices)[train_targets == cls]
                if len(cls_indices) > min_class_count:
                    cls_indices = resample(cls_indices, replace=False, n_samples=min_class_count, random_state=seed)
                balanced_train_indices.extend(cls_indices)

            balanced_train_indices = np.array(balanced_train_indices)
        
        elif method == 'smote':
            smote = SMOTE(random_state=seed)
            train_data = np.array(train_indices)
            balanced_train_indices, _ = smote.fit_resample(train_data.reshape(-1, 1), train_targets)
            balanced_train_indices = balanced_train_indices.flatten()
        
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        np.random.shuffle(balanced_train_indices)
        balanced_results.append((balanced_train_indices.tolist(), test_indices))

    return balanced_results


class FoldManagement: 
    """
    A class to manage the creation of the folds for trainning.

    Attributes:
    - method : str
        The fold method to use.
    - shuffle : bool
        Gives the user the choice to shuffle the data output and the fold creation.
    - random_state : int
        Seed used for the random creation.
    - train_aprox_size : float
        The balance of that between trinning and test datasets.

    Methods:
    - Split(x, y)
        Apply the selected labeling method to the provided metadata.
    """

    def __init__(self,
                 targets, 
                 method="kFold",
                 resampling_method="undersample", 
                 shuffle=False, 
                 random_state=42, 
                 balanced  = True,
                 train_aprox_size=0.8):
        self.targets = targets
        self.method = method
        self.resampling_method = resampling_method
        self.shuffle = shuffle
        self.seed = random_state
        self.train_aprox_size = train_aprox_size
        self.massives_count = {}
        self.results = None
        self.balanced = balanced

    def split(self, x, y):
        """
        Split the data into train and test sets based on the specified folding method.

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
                self.results = KFold_method(x, train_size=self.train_aprox_size, seed=self.seed, shuffle=self.shuffle)

            case "mFold":
                self.results = fold_massive_method(self.massives_count)

            case "combinationFold":
                self.results = combination_method(self.massives_count, train_size=self.train_aprox_size, proximity_value=1)
        
        if self.balanced:
            self.results = balance_classes(self.results, self.targets, method=self.resampling_method, seed=self.seed)

        return self.results
