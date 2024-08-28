import sys
import os
import time
import logging
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import joblib

# Get the parent directory and append it to the system path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import necessary libraries and modules
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from estimators.statistical_descriptor import Nagler_WS
from utils.dataset_load import shuffle_data, DatasetLoader
from utils.fold_management import FoldManagement
from utils.label_management import LabelManagement
from utils.balance_management import BalanceManagement
from utils.dataset_management import parse_pipeline
from utils.figures import *
from utils.files_management import *

def fit_predict_fold(pipeline, X_train_k, y_train_k, X_test_k, y_test_k, log_model, label_encoder, kfold, pipeline_name, save_dir, error_log_path):
    """
    Fit the pipeline on the training set and predict probabilities on the test set for a single fold.

    Parameters:
    - pipeline: The machine learning pipeline to be used.
    - X_train_k, y_train_k: Training data and labels for the fold.
    - X_test_k, y_test_k: Test data and labels for the fold.
    - log_model: Logger for model-related messages.
    - label_encoder: Label encoder for transforming labels.
    - kfold: The current fold index.
    - pipeline_name: Name of the pipeline.
    - save_dir: Directory to save the trained model.
    - error_log_path: Path to log errors.

    Returns:
    - fold_metric: Metrics for the current fold.
    - y_prob: Predicted probabilities for the test set.
    - y_test_k: True labels for the test set.
    """
    pipeline_id = f"{pipeline_name}_kfold_{kfold}"
    try:
        # Measure training time
        start_time = time.time()
        pipeline.fit(X_train_k, y_train_k)
        training_time = time.time() - start_time

        # Measure prediction time
        start_time = time.time()
        y_prob = pipeline.predict_proba(X_test_k)
        prediction_time = time.time() - start_time

        # Log prediction results and metrics
        log_model, fold_metric = report_prediction(log_model, y_test_k, y_prob, label_encoder, kfold)

        # Add timing information to fold metrics
        fold_metric["training_time"] = training_time
        fold_metric["prediction_time"] = prediction_time

        # Save the trained pipeline model
        joblib.dump(pipeline, os.path.join(save_dir, f"{pipeline_name}_fold{kfold}.joblib"))

        return fold_metric, y_prob, y_test_k
    except Exception as e:
        # Log any errors encountered during fitting or prediction
        error_message = f"Pipeline {pipeline_id} failed with error: {str(e)}"
        log_model.error(error_message)
        log_error_details(pipeline_id, str(e), error_log_path)
        return None, None, None

def predict_dataset(x, targets, fold_groups, pipeline_names, output_dir, args, label_encoder, error_log_path, rng, save=True):
    """
    Predict on the dataset using specified pipelines and folds.

    Parameters:
    - x: Feature data.
    - targets: Target labels.
    - fold_groups: Groups of training/testing indices for cross-validation.
    - pipeline_names: Names of the pipelines to evaluate.
    - output_dir: Directory to save results.
    - args: Command-line arguments.
    - label_encoder: Label encoder for targets.
    - error_log_path: Path to log errors.
    - rng: Random number generator for reproducibility.
    - save: Whether to save the metrics to disk.

    Returns:
    - metrics: Dictionary of metrics for each pipeline.
    - y_est_save: Dictionary of true and estimated labels.
    """
    y_est_save = {}
    metrics = {}
    f_tqdm = open(os.path.join(args.storage_path, 'progress.txt'), 'w')
    f_tqdm.write('tqdm\n')

    # Loop over each pipeline and process it
    for count, pipeline_name in enumerate(tqdm(pipeline_names, file=f_tqdm)):
        try:
            save_dir = os.path.join(output_dir, f"models/{pipeline_name}/")
            check_and_create_directory(save_dir)
            log_model, _ = init_logger(save_dir, f"{pipeline_name}_results")

            log_model.info(f"================== Fitting model {pipeline_name} ==================")

            y_est_save[pipeline_name] = {"y_true": [], "y_est": []}
            fold_metrics = []
            print(f"pipeline : {args.pipeline}")
            print(f"list of imports  : {args.import_list}")

            # Wrap fit and predict for parallel processing
            def fit_predict_fold_wrap(fold, train_index, test_index, rng):
                X_train_k, y_train_k = x[train_index], targets[train_index]
                X_test_k, y_test_k = x[test_index], targets[test_index]

                return fit_predict_fold(
                    parse_pipeline(args=args, idx=count, rng=rng),
                    X_train_k, y_train_k,
                    X_test_k, y_test_k,
                    log_model, 
                    label_encoder,
                    fold,
                    pipeline_name,
                    save_dir,
                    error_log_path
                )

            # Parallel processing for each fold
            results = Parallel(n_jobs=-1)(
                delayed(fit_predict_fold_wrap)(fold, train_index, test_index, rng)
                for fold, (train_index, test_index) in enumerate(fold_groups)
            )

            # Collect results from all folds
            for fold_metric, y_prob, y_test_k in results:
                if fold_metric is not None:
                    fold_metrics.append(fold_metric)
                    y_est_save[pipeline_name]["y_est"].extend(y_prob)
                    y_est_save[pipeline_name]["y_true"].extend(y_test_k)

            log_model = save_metrics(log_model, fold_metrics, pipeline_name)

            if save:
                dump_pkl(fold_metrics, os.path.join(save_dir, "metrics.pkl"))        
        
            metrics[pipeline_name] = fold_metrics

        except Exception as e:
            # Log any errors encountered during the pipeline processing
            error_message = f"Error occurred during model fitting and prediction for pipeline {pipeline_name}: {str(e)}"
            log_model.error(error_message)
            log_error_details(pipeline_name, str(e), error_log_path)

    return metrics, y_est_save

def log_error_details(pipeline_id, error_message, error_log_path):
    """
    Log error details to a specified error log file.

    Parameters:
    - pipeline_id: Identifier for the pipeline where the error occurred.
    - error_message: The error message to log.
    - error_log_path: Path to the log file for errors.
    """
    with open(error_log_path, "a") as log_file:
        log_file.write(f"{pipeline_id}: {error_message}\n")

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pipeline for validating and benchmarking machine learning models for wet snow characterization through imaging.')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--storage_path', type=str, required=True, help='Path to store the results')
    parser.add_argument('--fold_method', type=str, required=True, help='Method to fold the data')
    parser.add_argument('--labeling_method', type=str, required=True, help='Method to label the data')
    parser.add_argument('--balancing_method', type=str, required=True, help='Method to resample the data')
    parser.add_argument('--request', type=str, required=True, help='Request string to filter data')
    parser.add_argument('--shuffle_data', type=str, choices=['true', 'True', 'false', 'False'], required=True, help='Shuffle data or not')
    parser.add_argument('--balance_data', type=str, choices=['true', 'True', 'false', 'False'], required=True, help='Balance data or not')    
    parser.add_argument('--import_list', type=str, nargs='+', action='extend', required=True, help='List of imports')
    parser.add_argument('--pipeline', type=str, nargs='+', action='extend', required=True, help='Pipeline configurations')
    parser.add_argument('--metrics_to_report', type=str, nargs='+', action='extend', required=True, help='List of metrics to report')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()

    try:
        # Extract arguments for later use
        data_path = args.data_path
        storage_path = args.storage_path
        fold_method = args.fold_method
        seed = args.seed
        labeling_method = args.labeling_method
        balancing_method = args.balancing_method
        balance_data = args.balance_data.lower() == 'true'
        request = args.request
        shuffle_data = args.shuffle_data.lower() == 'true'
        metrics_to_report = args.metrics_to_report

    except KeyError as e:
        print("KeyError: %s undefined" % e)

    pipelines = []
    pipe = args.pipeline[0]
    for item in args.pipeline[1:]:
        if item.startswith('[[') and item.endswith('],'):
            pipelines.append(pipe)
            pipe = item
        else : 
            pipe = pipe +item

    # list_import = []
    # import_item = args.import_list[0]
    # for item in args.import_list[1:]:
    #     if item.startswith('from') :
    #         list_import.append(import_item)
    #         import_item = item
    #     else : 
    #         import_item = import_item + " " + item

    args.pipeline = pipelines 
    # args.import_list = list_import

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed=seed)
    np.random.seed(seed=seed)
    try:
        # Set up storage paths and initialize logging
        storage_path, pipeline_names = set_folder(storage_path, args=args)
        log_dataset, _ = init_logger(storage_path, "dataset_info")
        log_results, _ = init_logger(storage_path + "results", "results")
        log_errors, error_log_path = init_logger(storage_path + "results", "errors")

        # Load the dataset with specified parameters
        dataset_loader = DatasetLoader(
            data_path,
            shuffle=shuffle_data,
            descrp=[
                "date",
                "massif",
                "acquisition",
                "elevation",
                "slope",
                "orientation",
                "tmin",
                "hsnow",
                "tel"
            ],
            print_info=True,
            seed=rng
        )

        # Request specific data based on criteria
        # x, y = dataset_loader.request_data(request)
        
        # Manage labels according to the specified labeling method
        labels_manager = LabelManagement(method=labeling_method)
        targets = labels_manager.transform(y)
        label_encoder = labels_manager.get_encoder()
        
        # Manage folds for cross-validation
        fold_manager = FoldManagement(method=fold_method, shuffle=shuffle_data, rng=rng, train_aprox_size=0.8)
        fold_groups = fold_manager.split(x, y)
        
        # Balance data if specified
        if balance_data:
            balance_manager = BalanceManagement(method=balancing_method, rng=rng)
            fold_groups = balance_manager.transform(folds=fold_groups, targets=targets)
        
        # Log dataset information
        log_dataset = logger_dataset(log_dataset, x, y, label_encoder.inverse_transform(targets))
        log_dataset, fold_key = logger_fold(log_dataset, fold_groups, label_encoder.inverse_transform(targets), y)

        # Execute the prediction process across folds and pipelines
        metrics, y_est_save = predict_dataset(x=x,
                                    targets=targets,
                                    fold_groups=fold_groups,
                                    pipeline_names=pipeline_names,
                                    output_dir=storage_path,
                                    args=args,
                                    label_encoder=label_encoder,
                                    error_log_path=error_log_path,
                                    rng=rng,
                                    save=True)

        # Save fold keys and predictions
        dump_pkl(fold_key, os.path.join(os.path.join(storage_path, "results/"), "fold_key.pkl"))     
        dump_pkl(y_est_save, os.path.join(os.path.join(storage_path, "results/"), "results_y_est.pkl"))     

        # Prepare to save plots of the results
        results_dir_figures = os.path.join(storage_path, "results/plots/")

        metrics_to_plot = ["f1_score_macro", "f1_score_weighted", "f1_score_multiclass", "kappa_score", "training_time", "prediction_time"]
    
        # Generate plots for the specified metrics
        plot_boxplots(metrics, metrics_to_plot=metrics_to_plot, save_dir=results_dir_figures, fold_key=fold_key, labels_massives=(fold_method=="mFold"))
        plot_roc_curves(metrics, save_dir=results_dir_figures)

        # Report metrics in the results log
        log_results = report_metric_from_log(log_results, metrics, metrics_to_report)

        print("================== End of the study ==================")

    except Exception as e:
        # Log any unexpected errors
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        log_errors.error(error_message)
