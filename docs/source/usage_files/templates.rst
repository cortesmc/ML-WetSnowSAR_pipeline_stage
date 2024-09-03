.. _templates:

Examples of Configuration Templates
=====================

Below are examples of templates for the config_param.yml file.


Single Experiment with Uniform Seeds
--------------------------------------
This YAML configuration is designed for a single run with all models using the same seed.

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


Two Experiments with Different Seeds
------------------------------------
This YAML configuration supports two runs for all models, each using different seeds.


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
                "--seed": 42
             - options:
                # Random seed for reproducibility
                "--seed": 43


Two Experiments with Uniform Seeds but Separate Models
------------------------------------------------------
This YAML configuration enables two runs with the same seed but separates the models. This approach is useful for parallel processing. If you wish to compare models using only one seed, you can separate the testing and then gather the results through an action.

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

            # Random seed for reproducibility
                "--seed": 42
    varying_args:
        groups:
            - options:
                
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

            - options:
            # Configuration for each pipeline
            "--pipeline":
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


Multiple Experiments with Varied Seeds Across All Models
--------------------------------------------------------
This YAML configuration generates 10 runs with all models, allowing you to test model variability. It utilizes the range function to generate a series of float values.


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
        range:
            - options:
                # Random seed for reproducibility
                "--seed": 1,10,1
