import numpy as np
from sklearn.preprocessing import LabelEncoder

def crocus_methode(metadata):
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
    if physics_data is None:
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
        0 indicates Condition A is met,
        1 indicates Condition B is met,
        2 indicates Condition C is met.

    Raises:
    - ValueError
        If the 'physics' key is not present in the metadata dictionary.
    """
    physics_data = metadata.get('physics', None)
    if physics_data is None:
        raise ValueError("The dictionary does not contain a 'physics' key.")

    # Define the conditions for each label
    condition_a = (physics_data[:, 0] <= -1)
    condition_b = (physics_data[:, 0] > -1) & (physics_data[:, 2] < 1)
    condition_c = (physics_data[:, 0] > -1) & (physics_data[:, 2] >= 1)

    # Initialize labels array with default value (e.g., -1 for undefined)
    labels = np.full(physics_data.shape[0], -1)

    # Apply conditions to set labels
    labels[condition_a] = 0
    labels[condition_b] = 1
    labels[condition_c] = 2

    return labels

class label_management:
    """
    A class to manage labeling methods.

    Attributes:
    - methode : str
        The labeling method to use.

    Methods:
    - transform(metadata)
        Apply the selected labeling method to the provided metadata.
    """

    def __init__(self, methode):
        """
        Initialize the label_management class with a specified method.

        Parameters:
        - methode : str
            The labeling method to use. Currently supports "crocus".
        """
        self.methode = methode

    def transform(self, metadata):
        """
        Apply the selected labeling method to the provided metadata.

        Parameters:
        - metadata : dict
            A dictionary containing metadata.

        Returns:
        - labels : ndarray or None
            An array of labels if a supported method is selected; None otherwise.
        """
        match self.methode:
            case "crocus":
                labels =crocus_method_three_labels(metadata)
            case _:
                labels = None
            
        self.encoder = LabelEncoder()
        self.encoder.fit(labels)
        labels_encoded = self.encoder.transform(labels)
        
        return labels_encoded

    def get_encoder(self):
        return self.encoder