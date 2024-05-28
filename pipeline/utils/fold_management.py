import numpy as np
from sklearn.model_selection import KFold
import itertools

def KFold_methode(x, train_size=0.8, seed=None, shuffle=False):
    """
    Perform K-Fold cross-validation on the data.
    
    ::warning:: Add logger to save information for each fold for information porposees.
    
    Parameters:
    - x : array-like
        The data to split into K folds.
    - number_groups : int, optional (default=5)
        The number of folds. Default is 5.
    - seed : int or None, optional (default=None)
        Seed used by the random number generator for reproducibility. If None, a random seed will be selected.
    - shuffle : bool, optional (default=False)
        Whether to shuffle the data before splitting into folds.

    Returns:
    - generator
        A generator yielding indices for train and test sets for each fold.
    """
    if shuffle == False:
        seed = None

    n_splits = int(1/(1-train_size))
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=shuffle)        
    return kf.split(x)

def fold_massive_methode(dict_massives):
    """
    Generate train and test indices for each massive in a dictionary.

    ::warning:: Add logger to save information for each fold for information porposees.
    ::warning:: Add function balance, need of labels for the balance and sampling correction (undersampling / oversampling ).
    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.

    Yields:
    - tuple
        A tuple containing train and test indices for each massive.
    """
    unique_massives = list(dict_massives.keys())
    for i in range(len(unique_massives)):
        test_massive = unique_massives[i]
        train_indices = []
        test_indices = dict_massives[test_massive]['indices']
        for j in range(len(unique_massives)):
            if j != i:
                train_indices.extend(dict_massives[unique_massives[j]]['indices'])
        
        ratio_train = len(train_indices)/(len(train_indices) + len(test_indices))
        if ratio_train > 0.9 :
            # train_indices = balance(train_indices)
            pass
        
        yield train_indices, test_indices

def combination_methode(dict_massives):
    """
    Generate combinations of massives based on a given dictionary.

    ::warning:: Add logger to save information for each fold for information porposees.

    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.

    Yields:
    - tuple
        A tuple containing train and test indices for each valid combination of massives.
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
        if 75 <= percentage <= 85:
            valid_combinations.append(combo)

    for combo in valid_combinations:
        train_indices = []
        test_indices = []
        for massif in massives:
            if massif in combo:
                train_indices.extend(dict_massives[massif]['indices'])
            else:
                test_indices.extend(dict_massives[massif]['indices'])
        yield train_indices, test_indices

class fold_management: 

    def __init__(self, methode = "kfold" , shuffle=False, random_state=42, train_aprox_size=0.80):
        self.methode = methode
        self.shuffle = shuffle
        self.seed = random_state
        self.rng = np.random.default_rng(self.seed)
        self.train_aprox_size = train_aprox_size

    def split(self, x, y):
        massives_count = {}

        for index, name in enumerate(y['metadata'][:, 1]):
            if name not in massives_count:
                massives_count[name] = {'count': 0, 'indices': []}
            massives_count[name]['count'] += 1
            massives_count[name]['indices'].append(index)
            
        if ((np.unique(y['metadata'][:, 1]).size == 1) and (self.methode != "kFold")):
            self.methode = "kFold"
            return self.split(x=x, y=y)
        
        match self.methode: 
            case "kFold":
                return KFold_methode(x, train_size=self.train_aprox_size, seed=self.seed, shuffle=self.shuffle)
            
            case "mFold":
                return fold_massive_methode(massives_count)

            case "combinationFold":
                return combination_methode(massives_count)

            case _:
                return None
