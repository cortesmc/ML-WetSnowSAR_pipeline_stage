========================
Config_param Configuration
========================
The YAML file used for setting up an experiment is divided into two main sections: `fixed_args` and `varying_args`. Each section plays a specific role in defining the parameters and configurations for the experiment. This structured approach allows for efficient management of multiple experiments by specifying consistent parameters and introducing variability as needed.

Configuration Overview
-----------------------

Here is the YAML configuration used for defining the experiment parameters:

.. code-block:: yaml

    fixed_args:
        options:
            # Path to the dataset file
            "--data_path": "./data/dataset/dataset_AD_08200821_14Mas3Top3Phy_W15_corrected_V2.h5"
            
            # Method for the fold: simpleSplit, kFold, mFold, or combinationFold
            "--fold_method": "mFold"
            
            # Method for labeling: crocus, 3labels
            "--labeling_method": "crocus"
            
            # Method for resampling: bFold (can create more than 1000 folds, use with caution :D), oversample, undersample, or smote
            "--balancing_method": "undersample"
            
            # Filter request based on date and elevation criteria
            "--request": "'(date.dt.month == 3 and date.dt.day == 1) and ((elevation > 1000) and (elevation < 2000))'"
            
            # Shuffle the data before processing, shuffle uses the seed to reproduce the results
            "--shuffle_data": true
            
            # Balance the data
            "--balance_data": true
            
            # Import necessary libraries and modules
            "--import_list": 
                - "from sklearn.svm import SVC"
                - "from sklearn.neighbors import KNeighborsClassifier"
                - "from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier"
                - "from sklearn.linear_model import LogisticRegression"
                - "from sklearn.neural_network import MLPClassifier"
                - "from estimators.statistical_descriptor import *"
                - "from estimators.band_transform import *"
            
            # Configuration for each pipeline
            "--pipeline":
            # KNeighborsClassifier pipeline
              - - - "KNN_direct"
                - - "BandSelector"
                  - bands: [0,1,2,3,4,5,6,7,8]
                - - "BandTransformer"
                  - bands : [0,1,2,3]
                  - transformations : [] 
                - - "Hist_SAR"
                - - "KNeighborsClassifier"
                  - n_neighbors: 50
              # RandomForestClassifier pipeline
              - - - "RandomForest_direct"
                - - "BandSelector"
                  - bands: [0,1,2,3,4,5,6,7,8]
                - - "BandTransformer"
                  - bands : [0,1,2,3]
                  - transformations : [] 
                - - "Hist_SAR"
                - - "RandomForestClassifier"
                  - n_estimators: 200
                  - criterion: "entropy"
              # MLPClassifier pipeline
              - - - "MLP_direct"
                - - "BandSelector"
                  - bands: [0,1,2,3,4,5,6,7,8]
                - - "BandTransformer"
                  - bands : [0,1,2,3]
                  - transformations : [] 
                - - "Hist_SAR"
                - - "MLPClassifier"
                  - alpha: 0.01
              # LogisticRegression pipeline
              - - - "LogisticR_direct"
                - - "BandSelector"
                  - bands: [0,1,2,3,4,5,6,7,8]
                - - "BandTransformer"
                  - bands : [0,1,2,3]
                  - transformations : [] 
                - - "Hist_SAR"
                - - "LogisticRegression"
              # AdaBoostClassifier pipeline
              - - - "AdaBoost_direct"
                - - "BandSelector"
                  - bands: [0,1,2,3,4,5,6,7,8]
                - - "BandTransformer"
                  - bands : [0,1,2,3]
                  - transformations : [] 
                - - "Hist_SAR"
                - - "AdaBoostClassifier"
                  - n_estimators: 200   

            # Metrics to include in the final report
            "--metrics_to_report":
                - "f1_score_macro"
                - "f1_score_weighted"
                - "f1_score_multiclass"
                - "accuracy_score"
                - "precision_score_macro"
                - "recall_score_macro"
                - "roc_auc_score"
                - "log_loss"
                - "kappa_score"
                - "confusion_matrix"

    varying_args:
        groups:
            - options:
                # Random seed for reproducibility
                "--seed": 43

            - options:
                # Random seed for reproducibility
                "--seed": 42

Explanation of Fields
----------------------

fixed_args
^^^^^^^^^^

- **--data_path**: ``"./data/dataset/dataset_AD_08200821_14Mas3Top3Phy_W15_corrected_V2.h5"``  
  The file path to the dataset used in the experiment. Ensure this path is accurate and the dataset file exists at the specified location.

- **--fold_method**: ``"mFold"``  
  The method used to split the dataset into folds for cross-validation. Options include `simpleSplit`, `kFold`, `mFold`, and `combinationFold`.

- **--labeling_method**: ``"crocus"``  
  The method used for labeling the data. Options include `crocus` and `3labels`.

- **--balancing_method**: ``"undersample"``  
  The method used for resampling the data to balance classes. Options include `bFold`, `oversample`, `undersample`, and `smote`.

- **--request**: ``"'(date.dt.month == 3 and date.dt.day == 1) and ((elevation > 1000) and (elevation < 2000))'"``  
  A filter query for selecting data based on date and elevation criteria. Ensure the query matches the format expected by your data processing system.

- **--shuffle_data**: ``true``  
  Indicates whether to shuffle the data before processing. Setting this to `true` ensures that the data is randomized, which can help in producing more generalized models.

- **--balance_data**: ``true``  
  Indicates whether to apply balancing to the data. This should match the balancing method specified.

- **--import_list**:  
  A list of necessary imports for the experiment. Ensure all listed modules are installed and available in the environment where the experiment will be executed.

- **--pipeline**:  
  Defines the different machine learning pipelines to be used. Each pipeline consists of a sequence of processing steps and model configurations.

- **--metrics_to_report**:  
  A list of metrics that will be included in the final evaluation report. These metrics help assess the performance of the models used in the experiment.

varying_args
^^^^^^^^^^^^^

- **groups**:  
  This section allows you to define different experimental setups by varying certain parameters, such as the random seed. Each group represents a distinct configuration of the experiment.

  - **--seed**:  
    Specifies the random seed for reproducibility. Different seeds can be used to explore variations in model performance due to randomness in data splitting or initialization.

.. note::
    Another option for defining variations is the **range** method. This allows you to create multiple groups by specifying a range of values in a Python-like syntax. For example:

    .. code-block:: yaml

        varying_args:
            range:
                options:
                    --seed: [1, 10, 2]
    
    In this example, `--seed` will be set to 1, 3, 5, ...,9, creating multiple experimental setups with different seeds within the specified range.

Usage Notes
-----------

- Verify that all paths specified in the configuration file are correct and point to existing files or directories.
- Ensure that the Python environment has all the necessary libraries and modules as listed in `--import_list`.
- Adjust the `--pipeline` configurations to match the specific needs of your experiment. Add or remove steps as required.
- Use the `varying_args` section to create multiple experimental configurations by modifying parameters such as the random seed.
- The metrics in `--metrics_to_report` can be customized based on the evaluation criteria you wish to include in your final report.

