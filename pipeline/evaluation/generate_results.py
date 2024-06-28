import sys, os, argparse
import numpy as np 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.files_management import *
from utils.figures import *

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    storage_path = args.storage_path
    
    all_items = os.listdir(storage_path)

    yaml_file_path = None
    groups = []

    for item in all_items:
        item_path = os.path.join(storage_path, item)
        if os.path.isfile(item_path) and item.endswith('.yaml'):
            yaml_file_path = item_path
        else:
            groups.append(item_path)

    methods_param = load_yaml(yaml_file_path)
    print(groups[0]+"/results/fold_key.h5")
    fold_key = load_h5(groups[0]+"/results/fold_key.h5")

    metrics = {}
    for inx, group in enumerate(groups):
        models = [methods_param["groups_of_parameters"][inx]["--pipeline"][i][0][0] for i in range(len(x["groups_of_parameters"][0]["--pipeline"]))] 

        for model in models:
            metrics[model] = load_h5(group+"/models/"+model+"/metrics.h5")

    log_results, _ = init_logger(os.path.join(storage_path, "results"), "results")

    results_dir = os.path.join(os.path.dirname(storage_path), "results/plots/")

    metrics_to_plot = ["f1_score_macro", "f1_score_weighted", "f1_score_multiclass", "kappa_score", "training_time", "prediction_time"]

    plot_boxplots(metrics, metrics_to_plot=metrics_to_plot, save_dir=results_dir, fold_key=fold_key, labels_massives=(methods_param["groups_of_parameters"][0]["--fold_method"]=="mFold"))
    plot_roc_curves(metrics, save_dir=results_dir)

    log_results = report_metric_from_log(log_results, metrics, methods_param["groups_of_parameters"][0]["--metrics_to_report"])