import numpy as np

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
    
    condition = ((physics_data[:, 0] > 0) & (physics_data[:, 2] >= 40))
    
    labels = np.where(condition, 0, 1)
        
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
                return crocus_methode(metadata)
            case _:
                return None
