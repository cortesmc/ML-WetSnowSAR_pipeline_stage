import sys, os, argparse
import numpy as np 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.files_management import *
from utils.figures import *
from utils.dask_chunk import *

if __name__ == "__main__":
    metric_to_compare = "accuracy_score" #if there are multiple models trainned and tested on the same differen seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    try:
        config_file = args.config_file
        storage_path = args.storage_path

        config_param = load_yaml(config_file)

        massif_test = config_param["massif_test"]
        results_type = config_param["results_type"] 
        dates = config_param["dates"] 
        models_to_test = config_param["models"] 

    except KeyError as e:
        print("KeyError: %s undefined" % e)

    train_yaml = None
    folders = []
    if os.path.isdir(os.path.join(storage_path, 'group_0')):
        all_items = os.listdir(storage_path)
        for item in all_items:
            item_path = os.path.join(storage_path, item)
            if os.path.isfile(item_path) and item.endswith('.yaml'):
                train_yaml = item_path
            elif item.endswith('results_final'):
                continue
            elif item.endswith('results'):
                continue
            else:
                folders.append("./"+item_path)
    else:
        train_yaml = storage_path + "/info.yaml"
        folders = [storage_path]

    training_param = load_yaml(train_yaml)

    metrics = {}
    fold_key_test = []
    for idx, folder in enumerate(sorted(folders)):

        fold_dict= open_pkl(folder+"/results/fold_key.pkl")

        for key, value in fold_dict.items():
            train_set = value['train']
            if not all(name in train_set for name in massif_test):
                fold_key_test.append(key)
            models = [training_param["groups_of_parameters"][idx]["--pipeline"][i][0][0] for i in range(len(training_param["groups_of_parameters"][idx]["--pipeline"]))] 
       
        for model in models:
            try:
                if model not in metrics:
                    metrics[model] = []
                metric_tmp = open_pkl(folder+"/models/"+model+"/metrics.pkl")
                metric = []
                for dict_tmp in metric_tmp:
                    dict_tmp["group"] = idx
                    metric.append(dict_tmp)
                metrics[model] = metrics[model] + metric
            except Exception as e:
                continue

    print(metrics["KNN_direct"][15]["group"])

    pipelines = []
    for model in models_to_test:
        for fold in fold_key_test:
            index_pipeline = max((i for i in range(len(metrics[model])) if metrics[model][i]["fold"] == fold),key=lambda i: metrics[model][i]["accuracy_score"])
            group = metrics[model][index_pipeline]["group"]

            pipeline = joblib.load(storage_path+f"/group_{group}/models/{model}/{model}_fold{fold}.joblib")
            pipelines.append(pipeline)

    model = pipelines[0]
    
    print(model)

    # Create a sample 3D tensor
    tensor_3d = np.random.rand(100, 100, 9)

    print(tensor_3d.shape)
    transformer_custom = SlidingWindowTransformer(window_size=2, estimator=model, padding=False, use_predict_proba=True)


    # Transform 3D tensor
    result_custom_3d = transformer_custom.transform(tensor_3d)
    print(result_custom_3d)