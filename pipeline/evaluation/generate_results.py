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
    yaml_file_path = None
    folders = []
    if os.path.isdir(os.path.join(storage_path, 'group_0')):
        all_items = os.listdir(storage_path)
        for item in all_items:
            item_path = os.path.join(storage_path, item)
            if os.path.isfile(item_path) and item.endswith('.yaml'):
                yaml_file_path = item_path
            elif item.endswith('results'):
                continue
            else:
                folders.append("./"+item_path)
    else:
        yaml_file_path = storage_path + "/info.yaml"
        folders = [storage_path]

    methods_param = load_yaml(yaml_file_path)
    fold_key = open_pkl(folders[0]+"/results/fold_key.h5")
    
    metrics = {}
    
    for idx, folder in enumerate(sorted(folders)):
        models = [methods_param["groups_of_parameters"][idx]["--pipeline"][i][0][0] for i in range(len(methods_param["groups_of_parameters"][idx]["--pipeline"]))] 
        for model in models:
            try:
                if model not in metrics:
                    metrics[model] = []
                metrics[model] = metrics[model] + open_pkl(folder+"/models/"+model+"/metrics.h5")
            except Exception as e:
                continue

    check_and_create_directory(storage_path+"/results_final")
    log_results, _ = init_logger(os.path.join(storage_path, "results_final"), "results")

    results_dir_figures = os.path.join(storage_path, "results_final/plots/")

    metrics_to_plot = ["f1_score_macro", "f1_score_weighted", "f1_score_multiclass", "kappa_score", "training_time", "prediction_time"]

    plot_boxplots(metrics, metrics_to_plot=metrics_to_plot, save_dir=results_dir_figures, fold_key=fold_key, labels_massives=(methods_param["groups_of_parameters"][0]["--fold_method"]=="mFold"))
    plot_roc_curves(metrics, save_dir=results_dir_figures)

    log_results = report_metric_from_log(log_results, metrics, methods_param["groups_of_parameters"][0]["--metrics_to_report"])