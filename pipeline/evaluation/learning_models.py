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
from utils.files_management import (
    load_yaml,
    dump_pkl,
    init_logger,
    open_param_set_dir,
    report_prediction,
    report_metric_from_log,
    write_report,
)

#def Nagler_estimation(data_path):
def Nagler_estimation(X_trainU, y_train, X_test, y_test, label_encoder):
    y_est_save = {}
    '''X_trainU, y_train, label_encoder = load_train(
        data_path, -1, balanced=False, shffle=True, encode=True
    )
    X_test, y_test = load_test(
        data_path, -1, balanced=True, shffle=True, encoder=label_encoder
    )
    '''
    pos_class = label_encoder.transform(["wet"])[0]

    NGS_VV = Nagler_WS(bands=6)
    name_pip = "Nagler_VV"
    prob_test = NGS_VV.predict_proba(X_test)[:, pos_class]
    prob_train = NGS_VV.predict_proba(X_trainU)[:, pos_class]
    y_prob = np.concatenate([prob_train, prob_test])
    y_true = np.concatenate([y_train, y_test])

    y_est_save[name_pip] = {"y_true": y_true, "y_est": y_prob}

    NGS_VH = Nagler_WS(bands=7)
    name_pip = "Nagler_VH"
    prob_test = NGS_VH.predict_proba(X_test)[:, pos_class]
    prob_train = NGS_VH.predict_proba(X_trainU)[:, pos_class]
    y_prob = np.concatenate([prob_train, prob_test])
    y_true = np.concatenate([y_train, y_test])

    y_est_save[name_pip] = {"y_true": y_true, "y_est": y_prob}

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
    print(dtst_ld.infos)

    x, y = dtst_ld.request_data(request)


