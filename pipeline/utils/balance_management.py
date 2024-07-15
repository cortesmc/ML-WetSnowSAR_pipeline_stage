"""
BalanceManagement
=================

This module provides functionalities for balancing classes within folds using
various resampling methods, as well as creating balanced sub-folds for binary
or multi-label classification.

Other balancing methods can be added by creating a new function and adding the option to the balance_classes function with a new name.
The new balancing method must take a list of tuples and return a new list of tuples with training and test indices.
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import itertools
import random

def balance_classes(folds, targets, method='oversample', rng=42):
    """
    Balance the classes within each fold using the specified method.

    Parameters
    ----------
    folds : list of tuples
        A list containing train and test indices for each fold.
    targets : np.ndarray
        Target labels.
    method : str, optional
        The resampling method to use ('oversample', 'undersample', 'smote'). Default is 'oversample'.
    rng : int, optional
        rng for random number generator. Default is 42.

    Returns
    -------
    list of tuples
        A list containing balanced train and test indices for each fold.
    
    Raises
    ------
    ValueError
        If the specified resampling method is not recognized.
    """
    balanced_folds = []

    for train_indices, test_indices in folds:
        train_targets = targets[train_indices]
        
        if method == 'oversample':
            sampler = RandomOverSampler(random_state=rng)
            balanced_train_indices, _ = sampler.fit_resample(np.array(train_indices).reshape(-1, 1), train_targets)
            balanced_train_indices = balanced_train_indices.flatten()
        
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=rng)
            balanced_train_indices, _ = sampler.fit_resample(np.array(train_indices).reshape(-1, 1), train_targets)
            balanced_train_indices = balanced_train_indices.flatten()
        
        elif method == 'smote':
            sampler = SMOTE(random_state=rng)
            balanced_train_indices, _ = sampler.fit_resample(np.array(train_indices).reshape(-1, 1), train_targets)
            balanced_train_indices = balanced_train_indices.flatten()
        
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        balanced_folds.append((balanced_train_indices.tolist(), test_indices))

    return balanced_folds


def bFold(folds, targets, rng=42):
    """
    Create balanced sub-folds for binary or multi-label classification within each main fold.
    
    Parameters
    ----------
    folds : list of tuples
        A list containing train and test indices for each fold.
    targets : np.ndarray
        Target labels.
    rng : int, optional
        rng for random number generator. Default is 42.

    Returns
    -------
    list of tuples
        A list containing balanced train and test indices for each sub-fold.
    """
    sub_folds = []

    for train_indices, test_indices in folds:
        train_indices = np.array(train_indices)
        train_targets = targets[train_indices]

        # Identify the smallest class
        unique, counts = np.unique(train_targets, return_counts=True)
        smallest_class = unique[np.argmin(counts)]
        num_smallest_class = np.min(counts)

        # Store indices for each class and shuffle
        class_indices = {label: rng.permutation(train_indices[train_targets == label]) for label in unique}

        # Number of groups for each class based on smallest class size
        class_groups = {label: np.array_split(class_indices[label], max(1, len(class_indices[label]) // num_smallest_class)) for label in unique}

        # Generate sub-folds by combining groups
        smallest_class_groups = class_groups[smallest_class]
        del class_groups[smallest_class]
        
        other_class_groups = list(itertools.product(*class_groups.values()))

        for group in smallest_class_groups:
            for combo in other_class_groups:
                combined_train_indices = np.concatenate([group, *combo])
                rng.shuffle(combined_train_indices)
                sub_folds.append((combined_train_indices, test_indices))
    
    return sub_folds

def bFold_multiclass(folds, targets, rng=42):
    """
    Create balanced sub-folds for multi-label classification within each main fold.
    
    Parameters
    ----------
    folds : list of tuples
        A list containing train and test indices for each fold.
    targets : np.ndarray
        Target labels.
    rng : int, optional
        rng for random number generator. Default is 42.

    Returns
    -------
    list of tuples
        A list containing balanced train and test indices for each sub-fold.
    """
    sub_folds = []

    for train_indices, test_indices in folds:
        train_targets = targets[train_indices]

        unique, counts = np.unique(train_targets, return_counts=True)
        smallest_class = unique[np.argmin(counts)]
        smallest_class_indices = train_indices[train_targets == smallest_class]
        num_smallest_class = len(smallest_class_indices)

        # Store indices for each class
        class_indices = {label: train_indices[train_targets == label] for label in unique}

        # Shuffle the indices for each class
        for label in class_indices:
            rng.shuffle(class_indices[label])
        
        # Generate sub-folds by taking different subsets of each class
        num_sub_folds = min(len(indices) // num_smallest_class for indices in class_indices.values())
        for i in range(num_sub_folds):
            balanced_train_indices = np.concatenate([class_indices[label][i*num_smallest_class:(i+1)*num_smallest_class] for label in unique])
            rng.shuffle(balanced_train_indices)
            sub_folds.append((balanced_train_indices, test_indices))
    
    return sub_folds


class BalanceManagement: 
    def __init__(self, method=None, rng=None):
        """
        Initializes the BalanceManagement class with the specified method and rng.

        Parameters
        ----------
        method : str, optional
            The balancing method to use ('bFold', 'oversample', 'undersample', 'smote'). Default is None.
        rng : RandomState, optional
            rng for random number generator. Default is 42.
        """
        self.method = method
        self.rng = rng

    def transform(self, folds, targets):
        """
        Balances the provided folds according to the specified method.

        Parameters
        ----------
        folds : list of tuples
            A list containing train and test indices for each fold.
        targets : np.ndarray
            Target labels.

        Returns
        -------
        list of tuples
            A list containing balanced train and test indices for each fold.
        """
        balanced_folds = []
        if self.method == "bFold":
            balanced_folds = bFold(folds, targets, rng=self.rng)
        else:
            balanced_folds = balance_classes(folds, targets, method=self.method, rng=self.rng)
        
        return balanced_folds
