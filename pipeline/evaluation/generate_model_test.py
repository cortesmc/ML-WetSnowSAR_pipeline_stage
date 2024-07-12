import sys
import os
import argparse
from datetime import datetime
import numpy as np
import joblib
from osgeo import gdal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Custom utility imports
from utils.files_management import *
from utils.SlidingWindowTransformer import SlidingWindowTransformer

def convert_date_format(date_str):
    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
    return date_obj.strftime('%Y%m%d')

def load_config_and_params(config_file, storage_path):
    config_param = load_yaml(config_file)
    data_path = config_param["data_path"]
    file_names = config_param["file_names"]
    massif_test = config_param["massif_test"]
    results_type = config_param["results_type"]
    dates = config_param["dates"]
    models_to_test = config_param["models"]
    
    return data_path, file_names, massif_test, results_type, dates, models_to_test

def gather_training_info(storage_path):
    folders = []
    train_yaml = None

    if os.path.isdir(os.path.join(storage_path, 'group_0')):
        all_items = os.listdir(storage_path)
        for item in all_items:
            item_path = os.path.join(storage_path, item)
            if os.path.isfile(item_path) and item.endswith('.yaml'):
                train_yaml = item_path
            elif item not in ['results_final', 'qualitative_study', 'results']:
                folders.append(f"./{item_path}")
    else:
        train_yaml = os.path.join(storage_path, "info.yaml")
        folders = [storage_path]
    
    return train_yaml, folders

def get_models_and_metrics(folders, massif_test):
    metrics = {}
    fold_key_test = []
    
    for idx, folder in enumerate(sorted(folders)):
        fold_dict = open_pkl(os.path.join(folder, "results/fold_key.pkl"))
        for key, value in fold_dict.items():
            train_set = value['train']
            if not all(name in train_set for name in massif_test):
                fold_key_test.append(key)
            
            training_param = load_yaml(train_yaml)
            models = [training_param["groups_of_parameters"][idx]["--pipeline"][i][0][0] 
                      for i in range(len(training_param["groups_of_parameters"][idx]["--pipeline"]))]
            
            for model in models:
                if model not in metrics:
                    metrics[model] = []
                metric_tmp = open_pkl(os.path.join(folder, f"models/{model}/metrics.pkl"))
                for dict_tmp in metric_tmp:
                    dict_tmp["group"] = idx
                    metrics[model].append(dict_tmp)
    
    return metrics, fold_key_test

def load_pipelines(storage_path, metrics, models_to_test, fold_key_test, metric_to_compare):
    pipelines = []
    for model in models_to_test:
        possibles_pipelines = []
        scores =[]
        for fold in fold_key_test:
            index_pipeline = max(
                (i for i in range(len(metrics[model])) if metrics[model][i]["fold"] == fold),
                key=lambda i: metrics[model][i][metric_to_compare]
            )
            score = metrics[model][index_pipeline][metric_to_compare]
            group = metrics[model][index_pipeline]["group"]
            pipeline_path = os.path.join(storage_path, f"group_{group}/models/{model}/{model}_fold{fold}.joblib")
            
            scores.append(score)
            possibles_pipelines.append(pipeline_path)
        index_max_score = scores.index(max(scores))
        pipeline = joblib.load(possibles_pipelines[index_max_score])
        pipelines.append(pipeline)
    
    return pipelines

def load_images(data_path, file_names, massif_test, dates, log_errors):
    images = []
    if file_names[0] is not None:
        for name in file_names:
            dataset = gdal.Open(os.path.join(data_path, name))
            image_3d = dataset.ReadAsArray()
            image_3d = np.transpose(image_3d, (1, 2, 0))
            images.append(image_3d)
    else:
        log_errors.info("No file names provided.")
    
    if dates[0] is not None and massif_test[0] is not None:
        for massif in massif_test:
            for date in dates:
                dataset_path = os.path.join(data_path, f"{massif}_{date}.tif")
                dataset = gdal.Open(dataset_path)
                image_3d = dataset.ReadAsArray()
                image_3d = np.transpose(image_3d, (1, 2, 0))
                images.append(image_3d)
    else:
        log_errors.info("No dates provided.")
    
    return images

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    metric_to_compare = "accuracy_score"

    try:
        config_file = args.config_file
        storage_path = args.storage_path

        data_path, file_names, massif_test, results_type, dates, models_to_test = load_config_and_params(config_file, storage_path)

        new_file = check_and_create_directory(os.path.join(storage_path, "qualitative_study"))
        log_errors, error_log_path = init_logger(new_file, "errors")
        train_yaml, folders = gather_training_info(storage_path)
        
        metrics, fold_key_test = get_models_and_metrics(folders, massif_test)
        dates = [convert_date_format(date) for date in dates]
        pipelines = load_pipelines(storage_path, metrics, models_to_test, fold_key_test, metric_to_compare=metric_to_compare)
        model = pipelines[0]
        print(f"Number of models {len(pipelines)}")
        images = load_images(data_path, file_names, massif_test, dates, log_errors)
        print(f"Number of images loaded: {len(images)}")


        # Uncomment the following lines if the transformation and dumping are required
        for index, image in enumerate(images) : 
            transformer = SlidingWindowTransformer(estimator=model, window_size=15, padding=False, use_predict_proba=True)
            result_2d = transformer.transform(image)
            dump_pkl(result_2d, os.path.join(new_file, f"map_{index}.pkl"))

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        log_errors.error(error_message)
