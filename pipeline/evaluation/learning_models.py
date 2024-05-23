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

def evaluate_methods(log_F, data_path, bands_max, methods_param, y_nagler):
    X_trainU, y_train, label_encoder = load_train(
        data_path, bands_max, balanced=False, shffle=True, encode=True
    )
    X_test, y_test = load_test(
        data_path, bands_max, balanced=True, shffle=True, encoder=label_encoder
    )

    log_F = logg_info(
        log_F, X_trainU, y_train, X_test, y_test, label_encoder, bands_max
    )
    y_est = prediction_dataset(
        X_trainU,
        y_train,
        X_test,
        y_test,
        label_encoder,
        output_path,
        methods_param,
        log_F,
    )
    y_est.update(y_nagler)

    return y_est

def prediction_dataset(
    X_train,
    y_train,
    X_test,
    Y_test,
    label_encoder,
    output_dir,
    methods_param,
    logg,
    save=True,
):
    bkf = BFold(shuffle=False, random_state=42)

    kfold = 0
    y_est_save, metrics = {}, {}
    kappa, f1sc, acc = [], [], []
    pos_class = label_encoder.transform(["wet"])[0]

    for count in range(len(methods_param["pipeline"])):
        name_pip = methods_param["name_pip"][count]
        logg.info(f"Pipeline : {name_pip}")
        y_est_save[name_pip] = {"y_true": [], "y_est": []}

        for train_index in bkf.split(X_train, y_train):
            logg.info(f"Kfold : {kfold}")
            X_train_K, y_train_k = X_train[train_index], y_train[train_index]
            logg.info(f" y_train_k : {np.unique(y_train_k, return_counts=True)}")
            logg.info(f" X_train_K : {X_train_K.shape}")

            pipeline = parser_pipeline(methods_param, count)

            try:
                id_pip = name_pip + f"_kfold_{kfold}"
                pipeline.fit(X_train_K, y_train_k)

                y_prob = pipeline.predict_proba(X_test)[:, pos_class]

                logg, f1, ac, ka = report_prediction(
                    Y_test, y_prob, label_encoder, logg
                )
                f1sc.append(f1)
                acc.append(ac)
                kappa.append(ka)

                y_est_save[name_pip]["y_est"].extend(y_prob)
                y_est_save[name_pip]["y_true"].extend(Y_test)

            except Exception as e:
                logg.error(f"Pipeline {id_pip} failed")
                logg.error(e)
            kfold += 1
        metrics[name_pip] = {"f1": f1sc, "acc": acc, "kappa": kappa}
        logg = report_metric_from_log(metrics, logg)
        if save:
            dump_pkl(pipeline, os.path.join(output_dir, f"{name_pip}.pkl"))
            dump_pkl(metrics, os.path.join(output_dir, f"metrics.pkl"))
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

        request = data_param["request"]
        shuffle_data = data_param["shuffle_data"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)

    start_line = 0
    
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

    train_idx, test_idx = dtst_ld.split_train_test(test_size = 0.20)
    
    fold = fold_management(shuffle=True, random_state=42, test_size=0.2)
    train_index, test_index = fold.get_n_splits(train_data = x[train_idx], number_groups = 2)

    print(train_index, test_index)
