"""
LabelManagement
=================

This module provides functionalities for labeling data based on specific conditions,
including methods for single and multi-label classification.

Other labeling methods can be added by creating a new function and adding the option to the transform function with a new name.
The new labeling method method must take the dictionary with the metadata and return a new list of the labels.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

def crocus_method(metadata):
    """
    Apply the crocus method to label data based on specific conditions.

    Parameters:
    - metadata : dict
        A dictionary containing metadata. Must include a 'physics' key with corresponding data.

    Returns:
    - labels : ndarray
        An array of labels where 0 indicates the condition is met and 1 indicates the condition is not met.

    Raises:
    - ValueError
        If the 'physics' key is not present in the metadata dictionary.
    """
    physics_data = metadata.get('physics', None)
    if (physics_data is None):
        raise ValueError("The dictionary does not contain a 'physics' key.")
    
    condition = ((physics_data[:, 0] > 0) & (physics_data[:, 2] >= 0.40))
    
    labels = np.where(condition, 1, 0)
        
    return labels

def crocus_method_three_labels(metadata):
    """
    Apply the crocus method to label data based on specific conditions.

    Parameters:
    - metadata : dict
        A dictionary containing metadata. Must include a 'physics' key with corresponding data.

    Returns:
    - labels : ndarray
        An array of labels where:
        - "wet" indicates Condition A is met,
        - "kinda_wet" indicates Condition B is met,
        - "not_wet" indicates neither condition is met.

    Raises:
    - ValueError
        If the 'physics' key is not present in the metadata dictionary.
    """
    physics_data = metadata.get('physics', None)
    if physics_data is None:
        raise ValueError("The dictionary does not contain a 'physics' key.")

    condition_a = (physics_data[:, 0] <= -5)
    condition_b = (physics_data[:, 0] > -5) & (physics_data[:, 2] < 1)

    labels = np.full(physics_data.shape[0], '', dtype=object)

    labels[condition_a] = "wet"
    labels[condition_b] = "kinda_wet"

    unspecified_conditions = ~(condition_a | condition_b )
    labels[unspecified_conditions] = "not_wet" 

    return labels


class LabelManagement:
    def __init__(self, method):
        """
        Initialize the label_management class with a specified method.

        Parameters:
        - method : str
            The labeling method to use. Currently supports "crocus".
        """
        self.method = method
        self.encoder = LabelEncoder()

    def transform(self, metadata):
        """
        Apply the selected labeling method to the provided metadata.

        Parameters:
        - metadata : dict
            A dictionary containing metadata.

        Returns:
        - labels_encoded : ndarray
            An array of encoded labels.
        """
        match self.method:
            case "crocus":
                labels = crocus_method(metadata)
            case "3labels":
                labels = crocus_method_three_labels(metadata)
            case _:
                labels = np.array([])
        
        if "" in labels:
            print("Warning: Empty string found in labels. Replacing with 'Unknown'.")
            labels[labels == ""] = "Unknown"
        
        labels = labels.astype(str)
        self.encoder.fit(labels)
        labels_encoded = self.encoder.transform(labels)
        return labels_encoded

    def get_encoder(self):
        """Return the LabelEncoder instance."""
        return self.encoder
