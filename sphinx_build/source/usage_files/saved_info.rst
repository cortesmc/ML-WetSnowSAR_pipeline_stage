Saved Information of a Run
==========================

After running an experiment and applying actions, Qanat saves a variety of files and logs to document the process and results. The saved information is organized into several folders, each containing specific types of data.

Folder Structure
----------------

When a run is completed, a directory named after the run (e.g., `run_0`) is created. This directory contains the following key components:

- **`group_info.yaml`**: A YAML file that contains metadata about the experiment groups used in the run.
- **`info.yaml`**: A YAML file with general information about the run, such as configuration settings or parameters used.
- **`log_dataset_info.log`**: A log file that records details about the dataset used, such as preprocessing steps or data splits.

Models Directory (`models/`)
----------------------------

The `models/` folder stores the trained models and their associated logs:

- **Model Subfolders**: Each model (e.g., `AdaBoost_direct`, `KNN_direct`, etc.) has its own subfolder containing:
  - **Model Files (`*.joblib`)**: These are the serialized models saved after training, with different files corresponding to different cross-validation folds (e.g., `AdaBoost_direct_fold0.joblib`).
  - **`metrics.pkl`**: A pickled file containing the performance metrics of the model.
  - **`log_[Model]_results.log`**: A log file recording the results of the model's training and evaluation.

Results Directory (`results/`)
------------------------------

The `results/` folder includes detailed results and visualizations:

- **`fold_key.pkl`**: A pickled file containing the mapping between data folds and their corresponding indices.
- **Logs (`log_errors.log`, `log_results.log`)**: Logs that capture errors and general results of the run.
- **Plots (`plots/`)**: This subfolder contains visual representations of the results:
  - **`boxplot/`**: Boxplots for various metrics, such as F1 scores and training times, broken down by fold and model.
  - **`roc_curve/`**: ROC curves for different models, illustrating their performance in terms of true positive and false positive rates.

Final Results Directory (`results_final/`)
------------------------------------------

The `results_final/` folder aggregates and finalizes the results:

- **Logs and Plots**: Similar to the `results/` folder, but these are the consolidated results, potentially averaged across models with the same name.
- **Finalized Plots**: Boxplots and ROC curves that represent the final, aggregated performance metrics.

Other Logs and Output Files
---------------------------

- **`progress.txt`**: A file that tracks the progress of the experiment, useful for monitoring long-running processes.
- **`stderr.txt` and `stdout.txt`**: Standard error and output logs capturing any console output or errors during the experiment.

Example Folder Structure
------------------------

Below is an example of what the folder structure might look like:

.. code-block:: bash

   ── run_0
      ├── group_info.yaml
      ├── info.yaml
      ├── log_dataset_info.log
      ├── models/
      │   ├── AdaBoost_direct/
      │   │   ├── AdaBoost_direct_fold0.joblib
      │   │   ├── metrics.pkl
      │   │   ├── log_AdaBoost_direct_results.log
      │   ├── KNN_direct/
      │   │   ├── KNN_direct_fold0.joblib
      │   │   ├── metrics.pkl
      │   │   ├── log_KNN_direct_results.log
      │   └── RandomForest_direct/
      │       ├── RandomForest_direct_fold0.joblib
      │       ├── metrics.pkl
      │       ├── log_RandomForest_direct_results.log
      ├── progress.txt
      ├── results/
      │   ├── fold_key.pkl
      │   ├── log_results.log
      │   ├── plots/
      │   │   ├── boxplot/
      │   │   ├── roc_curve/
      ├── results_final/
      │   ├── log_results.log
      │   ├── plots/
      │       ├── boxplot/
      │       ├── roc_curve/
      ├── stderr.txt
      └── stdout.txt

This structure helps to organize the outputs of each run, making it easier to review the models, analyze the results, and debug any issues.
