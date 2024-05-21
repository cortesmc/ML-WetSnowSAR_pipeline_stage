import sys, os, time
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sklearn.preprocessing import LabelEncoder

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

def Nagler_estimation(data_path):
    y_est_save = {}
    X_trainU, y_train, label_encoder = load_train(
        data_path, -1, balanced=False, shffle=True, encode=True
    )
    X_test, y_test = load_test(
        data_path, -1, balanced=True, shffle=True, encoder=label_encoder
    )

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
    except KeyError as e:
        print("KeyError: %s undefine" % e)

    start_line = 0

    y_nagler = Nagler_estimation(data_path)

    print(y_nagler)
    
    dtst_ld = Dataset_loader(
        data_path,
        shuffle=False,
        descrp=[
            "date",
            "massif",
            "aquisition",
            "aquisition",
            "elevation",
            "slope",
            "orientation",
            "tmin",
            "hsnow",
            "tel",
        ],
    )
    print(dtst_ld.infos)

    # Example of request
    rq1 = "massif == 'VERCORS' and \
          ((date.dt.month == 3 and date.dt.day== 1) or \
          (elevation > 3000 and hsnow < 0.25))"

    rq2 = "massif == 'ARAVIS' & aquisition == 'ASC' & \
           elev == 900.0 & slope == 20 & theta == 45 "

    rq3 = "massif == 'ARAVIS' | date.dt.month == 1"

    x, y = dtst_ld.request_data(rq2)

    print(x)

