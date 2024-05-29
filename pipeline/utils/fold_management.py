import numpy as np
from sklearn.model_selection import KFold
import itertools

def KFold_method(x, train_size=0.8, seed=None, shuffle=False):
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

def fold_massive_method(dict_massives):
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

def combination_method(dict_massives, train_size=0.8, proximity_value=1):
    """
    Generate prioritized combinations of massives based on a given dictionary.

    Parameters:
    - dict_massives : dict
        A dictionary where keys are massives and values are dictionaries containing 'count' and 'indices' keys.

    Yields:
    - tuple
        A tuple containing train and test indices for each prioritized combination of massives.
    """

    total_count = sum(value['count'] for value in dict_massives.values())

    massives = list(dict_massives.keys())

    # Generate all possible combinations of massifs (excluding the empty combination)
    all_combinations = []
    for r in range(1, len(massives)):
        combinations_object = itertools.combinations(massives, r)
        combinations_list = list(combinations_object)
        all_combinations.extend(combinations_list)

    # Filter combinations to find valid ones that fall within the desired training size range
    valid_combinations = []
    for combo in all_combinations:
        combo_count = sum(dict_massives[massif]['count'] for massif in combo)
        percentage = (combo_count / total_count) * 100
        if (train_size * 100) - proximity_value <= percentage <= (train_size * 100) + proximity_value:
            valid_combinations.append(combo)

    valid_combinations.sort(key=lambda combo: len(massives) - len(combo))

    # Sets to keep track of uncovered massifs for training and testing
    uncovered_train_massives = set(massives)
    uncovered_test_massives = set(massives)

    # List to store the selected combinations
    selected_combinations = []
    
    # Greedily select combinations to cover all massifs in both training and test sets
    for combo in valid_combinations:
        if not uncovered_train_massives and not uncovered_test_massives:
            break  # Stop if all massifs are covered in both sets

        train_massifs_in_combo = set(combo)
        test_massifs_in_combo = set(massives) - train_massifs_in_combo

        # Select combination if it covers any remaining uncovered massifs in either set
        if uncovered_train_massives & train_massifs_in_combo or uncovered_test_massives & test_massifs_in_combo:
            selected_combinations.append(combo)
            uncovered_train_massives -= train_massifs_in_combo
            uncovered_test_massives -= test_massifs_in_combo

    # Generate train and test indices for each selected combination
    for combo in selected_combinations:
        train_indices = []
        test_indices = []
        for massif in massives:
            if massif in combo:
                train_indices.extend(dict_massives[massif]['indices'])
            else:
                test_indices.extend(dict_massives[massif]['indices'])
        yield train_indices, test_indices

class fold_management: 

    def __init__(self, 
                 methode = "kfold" , 
                 shuffle=False, 
                 random_state=42, 
                 train_aprox_size=0.8, 
                 logg=None
                 ):
        self.methode = methode
        self.shuffle = shuffle
        self.seed = random_state
        self.train_aprox_size = train_aprox_size
        self.massives_count = {}
        self.generator = None 
        self.logg = logg

    def split(self, x, y):
        for index, name in enumerate(y['metadata'][:, 1]):
            if name not in self.massives_count:
                self.massives_count[name] = {'count': 0, 'indices': []}
            self.massives_count[name]['count'] += 1
            self.massives_count[name]['indices'].append(index)
            
        if ((np.unique(y['metadata'][:, 1]).size == 1) and (self.methode != "kFold")):
            self.methode = "kFold"
            return self.split(x=x, y=y)
        
        match self.methode: 
            case "kFold":
                self.generator = KFold_method(x, train_size=self.train_aprox_size, seed=self.seed, shuffle=self.shuffle)

            case "mFold":
                self.generator = fold_massive_method(self.massives_count)

            case "combinationFold":
                self.generator = combination_method(self.massives_count, train_size=self.train_aprox_size, proximity_value=1)

        return self.generator
    
    def log_combinations(self, targets, metadata):
        """
        Log information about the combinations of train and test indices generated by combination_method.

        Parameters:
        - self.logg : logging.Logger
            The logger to use for logging the information.

        Returns:
        - None
        """
        
        gen_copy1, gen_copy2 = itertools.tee(self.generator)
        self.generator = gen_copy2
        for i, (train_indices, test_indices) in enumerate(gen_copy1):
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
            train_distribution = np.unique(targets[train_indices], return_counts=True)
            test_distribution = np.unique(targets[test_indices], return_counts=True)
            train_size = len(train_indices) / (len(train_indices) + len(test_indices))
            train_massives = np.unique(metadata["metadata"][train_indices,1])
            test_massives = np.unique(metadata["metadata"][test_indices,1])
            
            self.logg.info("############################################")
            self.logg.info(f"Fold {i + 1}:")
            self.logg.info(f"Number of Train Samples: {len(train_indices)}")
            self.logg.info(f"Number of Test Samples: {len(test_indices)}")
            self.logg.info(f"Distribution class train: {train_distribution}")
            self.logg.info(f"Distribution class test: {test_distribution}")
            self.logg.info(f"Train size: {train_size}")
            self.logg.info(f"Massives in train: {train_massives}")
            self.logg.info(f"Massives in test: {test_massives}")

