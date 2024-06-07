import sys, os, time
import numpy as np
from datetime import datetime


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from estimators.statistical_descriptor import Nagler_WS
# from plot.figure_roc import ROC_plot
from utils.dataset_management import load_train, load_test, parse_pipeline, BFold
from utils.dataset_load import save_h5_II, load_data_h5, load_info_h5, shuffle_data, DatasetLoader
from utils.fold_management import FoldManagement
from utils.label_management import LabelManagement
from utils.figures import plot_boxplots, plot_roc_curves
from utils.files_management import (
    load_yaml,
    dump_pkl,
    init_logger,
    open_param_set_dir,
    report_prediction,
    report_metric_from_log,
    write_report,
    set_folder,
    logger_dataset,
    logger_fold
)

def predict_dataset(
    x,
    targets,
    fold_groups,
    output_dir,
    pipeline_params,
    label_encoder,
    log_results,
    save=True
):
    y_est_save, metrics = {}, {}

    for count in range(len(pipeline_params["pipeline"])):
        pipeline_name = pipeline_params["pipeline_names"][count]
        save_dir = output_dir + f"models/{pipeline_name}/"
        log_model, path_log_model = init_logger(save_dir, f"{pipeline_name}_results")

        log_model.info(f"================== Fitting model {pipeline_name} ==================")

        y_est_save[pipeline_name] = {"y_true": [], "y_est": []}
        fold_metrics = []

        for kfold, (train_index, test_index) in enumerate(fold_groups):

            X_train_k, y_train_k = x[train_index], targets[train_index]
            X_test_k, y_test_k = x[test_index], targets[test_index]

            log_model.info(f"__________________ Fold {kfold} with {pipeline_name} __________________")

            pipeline = parse_pipeline(pipeline_params, count)

            try:
                pipeline_id = pipeline_name + f"_kfold_{kfold}"
                pipeline.fit(X_train_k, y_train_k)
                y_prob = pipeline.predict_proba(X_test_k)
                
                log_model, fold_metric = report_prediction(log_model, y_test_k, y_prob, label_encoder, kfold)
                fold_metrics.append(fold_metric)

                y_est_save[pipeline_name]["y_est"].extend(y_prob)
                y_est_save[pipeline_name]["y_true"].extend(y_test_k)
            except Exception as e:
                log_model.error(f"Pipeline {pipeline_id} failed")
                log_model.error(e)
                
            if save:
                dump_pkl(pipeline, os.path.join(save_dir, f"{pipeline_name}_fold{kfold}.pkl"))

        if save:
            dump_pkl(fold_metrics, os.path.join(save_dir, f"metrics.pkl"))
        metrics[pipeline_name] = fold_metrics

    plot_boxplots(metrics, save_dir=output_dir + "results/plots/")
    plot_roc_curves(metrics, save_dir=output_dir + "results/plots/")
    log_results = report_metric_from_log(log_results, metrics)

    return y_est_save

if __name__ == "__main__":
    
    param_path = "pipeline/parameter/config_pipeline.yml"
    pipeline_params = load_yaml(param_path)

    try:
        data_path = pipeline_params["data_path"]
        out_dir = pipeline_params["out_dir"]
        seed = pipeline_params["seed"]
        BANDS_MAX = pipeline_params["BANDS_MAX"]
        fold_method = pipeline_params["fold_method"]
        request = pipeline_params["request"]
        shuffle_data = pipeline_params["shuffle_data"]
    except KeyError as e:
        print("KeyError: %s undefined" % e)

    out_dir = set_folder(out_dir, pipeline_params)
    log_dataset, path_log_dataset = init_logger(out_dir, "dataset_info")
    log_results, path_log_results = init_logger(out_dir + "results", "results")
    
    start_line = 0

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
        print_info=True
    )

    x, y = dataset_loader.request_data(request)

    fold_manager = FoldManagement(method=pipeline_params["fold_method"], 
                                  shuffle=pipeline_params["shuffle_data"], 
                                  random_state=pipeline_params["seed"], 
                                  train_aprox_size=0.8)

    labels_manager = LabelManagement(method=pipeline_params["labeling_method"])

    fold_groups = fold_manager.split(x, y)

    targets = labels_manager.transform(y)
    label_encoder = labels_manager.get_encoder()

    log_dataset = logger_dataset(log_dataset, x, y, targets, pipeline_params)
    log_dataset = logger_fold(log_dataset, fold_groups, targets, y)

    y_est_save = predict_dataset(x=x,
                                 targets=targets,
                                 fold_groups=fold_groups,
                                 output_dir=out_dir,
                                 pipeline_params=pipeline_params,
                                 label_encoder=label_encoder,
                                 log_results=log_results,
                                 save=True)
    
    print("================== End of the study ==================")