import sys, os, time
import numpy as np
from datetime import datetime


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from estimators.statistical_descriptor import Nagler_WS
# from plot.figure_roc import ROC_plot
from utils.dataset_management import load_train, load_test, parser_pipeline, BFold
from utils.dataset_load import  save_h5_II, load_data_h5, load_info_h5, shuffle_data, Dataset_loader
from utils.fold_management import fold_management
from utils.label_management import label_management
from utils.figures import plot_boxplots,  plot_roc_curves
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

def prediction_dataset(
    x,
    targets,
    fold_groupes,
    output_dir,
    pipeline_param,
    label_encoder,
    log_results,
    save=True
):
    y_est_save, metrics = {}, {}

    for count in range(len(pipeline_param["pipeline"])):
        name_pip = pipeline_param["name_pip"][count]
        save_dir = output_dir + f"models/{name_pip}/"
        log_model, path_log_model = init_logger(save_dir, f"{name_pip}_results")

        log_model.info(f"================== Fitting model {name_pip} ==================")

        y_est_save[name_pip] = {"y_true": [], "y_est": []}
        fold_metrics = []

        for kfold, (train_index, test_index) in enumerate(fold_groupes):

            X_train_K, y_train_k = x[train_index], targets[train_index]
            X_test_K, y_test_k = x[test_index], targets[test_index]

            log_model.info(f"__________________ Fold {kfold} with {name_pip} __________________")

            pipeline = parser_pipeline(pipeline_param, count)

            try:
                id_pip = name_pip + f"_kfold_{kfold}"
                pipeline.fit(X_train_K, y_train_k)
                y_prob = pipeline.predict_proba(X_test_K)
                
                log_model, fold_metric = report_prediction(log_model, y_test_k, y_prob, label_encoder, kfold)
                fold_metrics.append(fold_metric)

                y_est_save[name_pip]["y_est"].extend(y_prob)
                y_est_save[name_pip]["y_true"].extend(y_test_k)
            except Exception as e:
                log_model.error(f"Pipeline {id_pip} failed")
                log_model.error(e)
                
            if save:
                dump_pkl(pipeline, os.path.join(save_dir, f"{name_pip}_fold{kfold}.pkl"))

        if save:
                dump_pkl(fold_metrics, os.path.join(save_dir, f"metrics.pkl"))
        metrics[name_pip] = fold_metrics

    plot_boxplots(metrics, save_dir=output_dir+"results/plots/")
    plot_roc_curves(metrics, save_dir=output_dir+"results/plots/")
    log_results = report_metric_from_log(log_results, metrics)

    return y_est_save

if __name__ == "__main__":
    # Pipeline variables de systeme
    param_path = "pipeline/parameter/config_pipeline.yml"
    pipeline_param = load_yaml(param_path)


    try:
        data_path = pipeline_param["data_path"]
        out_dir = pipeline_param["out_dir"]
        seed = pipeline_param["seed"]
        BANDS_MAX = pipeline_param["BANDS_MAX"]
        methode_fold = pipeline_param["methode_fold"]
        request = pipeline_param["request"]
        shuffle_data = pipeline_param["shuffle_data"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)

    out_dir = set_folder(out_dir, pipeline_param)
    log_dataset, path_log_dataset = init_logger(out_dir, "dataset_info")
    log_results, path_log_results = init_logger(out_dir+"results", "resultats")
    
    start_line = 0

    dtst_ld = Dataset_loader(
        data_path,
        shuffle=shuffle_data,
        descrp=[
            "date",
            "massif",
            "aquisition",
            "elevation",
            "slope",
            "orientation",
            "tmin",
            "hsnow",
            "tel"
        ],
        print_info = True
    )

    x, y = dtst_ld.request_data(request)

    fold_manager = fold_management(methode=pipeline_param["methode_fold"], 
                           shuffle=pipeline_param["shuffle_data"], 
                           random_state=pipeline_param["seed"], 
                           train_aprox_size=0.8, ## à verifier si on ajoute dans le .yml.
                           )

    labels_manager  = label_management(methode=pipeline_param["labeling_methode"])

    fold_groupes = fold_manager.split(x, y)

    targets = labels_manager.transform(y)
    label_encoder = labels_manager.get_encoder()

    log_dataset = logger_dataset(log_dataset, x, y, targets, pipeline_param)
    log_dataset = logger_fold(log_dataset, fold_groupes,targets, y)

    y_est_save = prediction_dataset(x=x,
                                    targets=targets,
                                    fold_groupes=fold_groupes,
                                    output_dir=out_dir,
                                    pipeline_param=pipeline_param,
                                    label_encoder=label_encoder,
                                    log_results=log_results,
                                    save=True
                                    )