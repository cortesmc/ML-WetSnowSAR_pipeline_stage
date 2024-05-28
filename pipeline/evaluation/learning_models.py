import sys, os, time
import numpy as np

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
from utils.files_management import (
    load_yaml,
    dump_pkl,
    init_logger,
    open_param_set_dir,
    report_prediction,
    report_metric_from_log,
    write_report,
)

def logg_info(log_F, X_trainU, y_train, X_test, y_test, label_encoder, bands_max):
    """Logg information about the dataset"""
    log_F.info("############################################")
    log_F.info(f"Loaded {X_trainU.shape} train samples and {X_test.shape} test samples")
    log_F.info(f"Y_train: {np.unique(y_train, return_counts=True)}")
    log_F.info(f"Y_test: {np.unique(y_test, return_counts=True)}")
    log_F.info(f"Classes {label_encoder.classes_}")
    log_F.info(f"Labels {label_encoder.transform(label_encoder.classes_)}")
    log_F.info(f"List of bands {bands_max}")
    log_F.info("############################################")
    return log_F


def prediction_dataset(
    x,
    y,
    # output_dir,
    pipeline_param,
    data_param,
    logg,
    save=True,
):
    fold_manager = fold_management(methode=pipeline_param["methode_fold"], 
                           shuffle=data_param["shuffle_data"], 
                           random_state=pipeline_param["seed"], 
                           train_aprox_size=0.8 ## à verifier si on ajoute dans le .yml.
                           )

    labels_manager  = label_management(methode = pipeline_param["labeling_methode"])

    fold_groupes = fold_manager.split(x, y)

    y_est_save, metrics = {}, {}
    kappa, f1sc, acc = [], [], []

    targets = labels_manager.transform(y)
    label_encoder = labels_manager.get_encoder()

    for count in range(len(pipeline_param["pipeline"])):
        name_pip = pipeline_param["name_pip"][count]
        logg.info(f"Pipeline : {name_pip}")
        y_est_save[name_pip] = {"y_true": [], "y_est": []}

        for kfold, (train_index, test_index) in enumerate(fold_groupes):
            logg.info(f"Kfold : {kfold}")
            X_train_K, y_train_k = x[train_index], targets[train_index]
            X_test_K, y_test_k = x[test_index], targets[test_index]
            logg.info(f" y_train_k : {np.unique(y_train_k, return_counts=True)}")
            logg.info(f" X_train_K : {X_train_K.shape}")

            pipeline = parser_pipeline(pipeline_param, count)

            try:
                id_pip = name_pip + f"_kfold_{kfold}"
                pipeline.fit(X_train_K, y_train_k)

                y_prob = pipeline.predict_proba(X_test_K)

                logg, f1, ac, ka = report_prediction(
                    y_test_k, y_prob, label_encoder, logg
                )

                print(y_test_k)
                f1sc.append(f1)
                acc.append(ac)
                kappa.append(ka)

                y_est_save[name_pip]["y_est"].extend(y_prob)
                y_est_save[name_pip]["y_true"].extend(y_test_k)

            except Exception as e:
                logg.error(f"Pipeline {id_pip} failed")
                logg.error(e)
            kfold += 1
        metrics[name_pip] = {"f1": f1sc, "acc": acc, "kappa": kappa}
        logg = report_metric_from_log(metrics, logg)
        if save:
            pass
            # dump_pkl(pipeline, os.path.join(output_dir, f"{name_pip}.pkl"))
            # dump_pkl(metrics, os.path.join(output_dir, f"metrics.pkl"))
    return y_est_save


if __name__ == "__main__":
    param_path = "pipeline/parameter/config_pipeline.yml"
    pipeline_param = load_yaml(param_path)

    match pipeline_param["type"]:
        case "local":
            local_param_path = "pipeline/parameter/config_data_local.yml"
            data_param = load_yaml(local_param_path)
        case "global":
            global_param_path = "pipeline/parameter/config_data_global.yml"
            data_param = load_yaml(global_param_path)
        case _:
            f"no such type : {pipeline_param["type"]}"

    try:
        data_path = pipeline_param["data_path"]
        out_dir = pipeline_param["out_dir"]
        seed = pipeline_param["seed"]
        BANDS_MAX = pipeline_param["BANDS_MAX"]
        methode_fold = pipeline_param["methode_fold"]

        request = data_param["request"]
        shuffle_data = data_param["shuffle_data"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)

    start_line = 0
    log_F, path_log = init_logger(out_dir)

    dtst_ld = Dataset_loader(
        data_path,
        shuffle=shuffle_data,
        descrp=[
            "date",
            "massif",
            "aquisition",
            "aquisition2",
            "elevation",
            "slope",
            "orientation",
            "tmin",
            "hsnow",
            "tel",
        ],
        print_info = True
    )

    x, y = dtst_ld.request_data(request)
    log_F.info(f"================== Fitting model {"tmptmptmp"} ==================")
    log_F, path_log = init_logger(out_dir)


    y_est_save = prediction_dataset(
                                    x, 
                                    y, 
                                    pipeline_param, 
                                    data_param, 
                                    log_F, 
                                    save=True,)