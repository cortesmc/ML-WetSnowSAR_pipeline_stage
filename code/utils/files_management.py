import re, os, h5py, logging, pickle, shutil, zipfile, yaml, joblib, json
import numpy as np
import pandas as pd
import ast
from yaml import safe_load
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, roc_curve, log_loss,
    f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
)


def dump_pkl(obj, path):
    """
    Dump object in pickle file.

    Parameters
    ----------
    obj : object
        Object to dump, can be a list, a dict, a numpy array, etc.
    path : str
        Path to the pickle file.

    Returns
    -------
    int
        1 if the dump is successful.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1

def dump_h5(data, file_path):
    """
    Dump data to an HDF5 file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to store.
    file_path : str
        The path to the HDF5 file.
    
    Returns
    -------
    None
    """
    data_dict = {}
    data_dict["data"] = data
    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, (list, dict)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = str(value)

    
def load_h5(file_path):
    """
    Load data from an HDF5 file.

    Parameters
    ----------
    file_path : str
        The path to the HDF5 file.

    Returns
    -------
    numpy.ndarray
        The loaded data.
    """
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data_dict[key] = np.array(f[key])
        for key in f.attrs:
            try:
                data_dict[key] = json.loads(f.attrs[key])
            except json.JSONDecodeError:
                data_dict[key] = f.attrs[key]
    return data_dict["data"]

def open_pkl(path):
    """
    Open pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file to open.

    Returns
    -------
    object
        Object contained in the pickle file.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj  

def open_log_file(path_log):
    """
    Open a log file (.log) and return its contents.

    Parameters
    ----------
    path_log : str
        The path to the log file.

    Returns
    -------
    list
        A list of lines from the log file.
    """
    with open(path_log, "r") as f:
        log = f.readlines()
    return log


def clean_log(log):
    """
    Clean a log file to remove extraneous information for better readability.

    Parameters
    ----------
    log : list
        The list of lines from the log file.

    Returns
    -------
    list
        The cleaned log lines.
    """
    result = []
    for i in range(len(log)):
        pattern = r"Line: \d+ - (.*)$"
        match = re.search(pattern, log[i])

        if match:
            result.append(match.group(1) + "\n")
        else:
            result.append(log[i])
    return result


def write_report(path_log, path_report, begin_line=0):
    """
    Write a txt report from a log file (.log).

    Parameters
    ----------
    path_log : str
        Path to the log file.
    path_report : str
        Path to the report file.
    begin_line : int, optional
        Line number to start the report, by default 0.

    Returns
    -------
    list
        List of lines in the log file without useless information.
    int
        Number of lines in the log file.
    """
    op = open_log_file(path_log)[begin_line:]
    result = clean_log(op)
    with open(path_report, "w") as f:
        f.writelines(result)
    return result, len(op)


def report_metric_from_log(logg, dic, metrics_to_report=["f1_score_weighted"]):
    """
    Report metrics from a dictionary of model scores.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance to log the metrics.
    dic : dict
        Dictionary containing metrics for each model.
    metrics_to_report : list, optional
        List of metrics to report, by default includes 'f1_score_weighted'.

    Returns
    -------
    logging.Logger
        The logger instance with reported metrics.
    """
    logg.info(f"================== Final report ==================")
    for model_name, model_metrics in dic.items():
        logg.info(f"__________________ Model : {model_name} __________________")

        for metric in metrics_to_report:
            try:
                if metric in model_metrics[0]:
                    values = [fold_metrics[metric] for fold_metrics in model_metrics]

                if isinstance(values[0], (int, float)):
                    logg.info(f"{metric} : {np.mean(values)} +/- {np.std(values)}")
                elif isinstance(values[0], np.ndarray):
                    values_flat = np.mean(values, axis=0)
                    logg.info(f"{metric} : {values_flat}")
                elif isinstance(values[0], pd.DataFrame):
                    mean_conf_matrix = sum(values) / len(values)
                    logg.info(f"{metric} :\n{mean_conf_matrix}")
                else:
                    logg.error(f"No suitable values found for metric {metric} in model {model_name}")
            except IndexError:
                logg.error(f"No results found for model {model_name}. Skipping metrics.")
                break
            except Exception as e:
                logg.error(f"Error reporting metric {metric} for model {model_name}: {e}")
            
    logg.info(f"================== End report ==================")
    return logg

def report_prediction(logg, y_true, y_pred, le, fold):
    """
    Compute and log various classification metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels (categorical or binary).
    y_pred : numpy.ndarray
        Predicted probabilities or classes.
    le : LabelEncoder
        LabelEncoder object for inverse transforming labels.
    logg : logging.Logger
        Logger instance for reporting.
    fold : int
        Fold number for identification.

    Returns
    -------
    tuple
        (Logger instance, dictionary of computed metrics).
    """  
    if y_pred.ndim > 1:
        if y_pred.shape[1] > 1:
            y_pred_classes = y_pred.argmax(axis=1)
        else:
            y_pred_classes = (y_pred[:, 0] >= 0.5).astype(int)
    else:
        y_pred_classes = (y_pred >= 0.5).astype(int)
    
    y_true_transformed = le.inverse_transform(y_true)
    y_pred_transformed = le.inverse_transform(y_pred_classes)

    all_labels = le.classes_

    cm = confusion_matrix(y_true_transformed, y_pred_transformed, labels=all_labels)
    cm_df = pd.DataFrame(
        100 * cm.astype(float) / cm.sum(axis=1, keepdims=True),
        columns=all_labels,
        index=all_labels
    ).round(4).fillna(0)

    f1_macro = 100 * round(f1_score(y_true_transformed, y_pred_transformed, average="macro"), 4)
    f1_weighted = 100 * round(f1_score(y_true_transformed, y_pred_transformed, average="weighted"), 4)
    f1_multiclass = 100 * np.round(f1_score(y_true_transformed, y_pred_transformed, average=None), 4)
    accuracy = 100 * round(accuracy_score(y_true_transformed, y_pred_transformed), 4)
    precision_macro = 100 * round(precision_score(y_true_transformed, y_pred_transformed, average="macro"), 4)
    recall_macro = 100 * round(recall_score(y_true_transformed, y_pred_transformed, average="macro"), 4)

    if y_pred.shape[1] > 2:
        roc_auc = 100 * round(roc_auc_score(y_true, y_pred, multi_class="ovr"), 4)
    else:
        roc_auc = 100 * round(roc_auc_score(y_true, y_pred_classes), 4)
    
    log_loss_val = 100 * round(log_loss(y_true, y_pred), 4)
    kappa = 100 * round(cohen_kappa_score(y_true_transformed, y_pred_transformed), 4)

    metrics = {
        'f1_score_macro': f1_macro,
        'f1_score_weighted': f1_weighted,
        'f1_score_multiclass': f1_multiclass,
        'accuracy_score': accuracy,
        'precision_score_macro': precision_macro,
        'recall_score_macro': recall_macro,
        'roc_auc_score': roc_auc,
        'log_loss': log_loss_val,
        'kappa_score': kappa,
        'confusion_matrix': cm_df
    }

    for metric, value in metrics.items():
        logg.info(f"{metric} : {value}")

    metrics['y_true'] = y_true_transformed
    metrics['y_pred'] = y_pred
    metrics['fold'] = fold

    return logg, metrics



def init_logger(path_log, name):
    """
    Initialize a logger for recording activities.

    Parameters
    ----------
    path_log : str
        Path to the log file.
    name : str
        Name of the logger instance.

    Returns
    -------
    tuple
        (Logger instance, path to the log file).
    """
    datestr = "%m/%d/%Y-%I:%M:%S %p"
    filename = os.path.join(path_log, f"log_{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s", datefmt=datestr)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.info("Started")
    return logger, filename

def logger_dataset(logg, x, metadata, targets):
    """
    Log detailed information about the dataset.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance for reporting.
    x : numpy.ndarray
        Input features.
    metadata : dict
        Metadata containing additional dataset information.
    targets : numpy.ndarray
        Target labels.

    Returns
    -------
    logging.Logger
        Logger instance with dataset information logged.
    """
    logg.info("================== Study information ==================")

    logg.info("__________________ Dataset information __________________")

    massives, counts = np.unique(metadata["metadata"][:, 1], return_counts=True)

    logg.info(f"Number of samples: {x.shape[0]}")
    logg.info(f"Dimensions of input: {x.shape[1:]}")
    
    overall_unique_targets, overall_target_counts = np.unique(targets, return_counts=True)
    overall_target_ratios = overall_target_counts / targets.size
    overall_target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(overall_unique_targets, overall_target_counts, overall_target_ratios))
    logg.info(f"Overall label distribution: {overall_target_info}")
    logg.info(f"Massives: {massives}")
    logg.info("Samples per massif:")

    for massive in massives:
        massive_indices = np.where(metadata["metadata"][:, 1] == massive)
        unique_targets, target_counts = np.unique(targets[massive_indices], return_counts=True)
        target_ratios = target_counts / target_counts.sum()
        target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(unique_targets, target_counts, target_ratios))
        
        logg.info(f"  {massive}: {counts[massives == massive][0]} samples (Targets: {target_info})")
        
    logg.info("__________________ Folds information __________________")

    return logg

def logger_fold(logg, fold_groups, targets, metadata):
    """
    Log information about each fold in cross-validation.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance.
    fold_groups : list of tuples
        List of (train_index, test_index) tuples for each fold.
    targets : numpy.ndarray
        Target labels.
    metadata : dict
        Metadata dictionary.

    Returns
    -------
    tuple
        (Logger instance, dictionary containing massif information for each fold).
    """
    fold_key = {}
    for kfold, (train_index, test_index) in enumerate(fold_groups):
        logg.info(f"------------------ Fold: {kfold} ------------------")
        
        train_unique_targets, train_target_counts = np.unique(targets[train_index], return_counts=True)
        train_target_ratios = train_target_counts / train_target_counts.sum()
        train_target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(train_unique_targets, train_target_counts, train_target_ratios))
        
        test_unique_targets, test_target_counts = np.unique(targets[test_index], return_counts=True)
        test_target_ratios = test_target_counts / test_target_counts.sum()
        test_target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(test_unique_targets, test_target_counts, test_target_ratios))
        
        massif_train = metadata['metadata'][train_index, 1]
        massif_test = metadata['metadata'][test_index, 1]

        logg.info(f"    - Distribution class train: {train_target_info}")
        logg.info(f"    - Distribution class test: {test_target_info}")
        logg.info(f"    - Train size: {len(train_index) / (len(train_index) + len(test_index)) * 100:.2f}%")
        logg.info(f"    - Massif in train: {np.unique(massif_train)}")
        logg.info(f"    - Massif in test: {np.unique(massif_test)}")

        fold_key[kfold] = {"train": np.unique(massif_train), "test": np.unique(massif_test)}

    return logg, fold_key

def save_metrics(log_model, fold_metric, model_name):
    """
    Log the metrics for each fold of the model.

    Parameters
    ----------
    log_model : logging.Logger
        Logger instance.
    fold_metric : list of dict
        List of metrics for each fold.
    model_name : str
        Name of the model.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    for metrics in fold_metric:
        log_model.info(f"__________________ Fold {str(metrics['fold'])} with {model_name} __________________")
        for metric, value in metrics.items():
            if metric in ["fold", "y_pred", "y_true"]:
                continue

            if isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, np.ndarray):
                value_str = "\n" + str(value)
            else:
                value_str = str(value)
            log_model.info(f" {metric} : {value_str}")

    return log_model


def save_h5(img, label, filename):
    """
    Save image and label in a hdf5 file.

    Parameters
    ----------
    img : numpy.ndarray
        Dataset of images in float32.
    label : numpy.ndarray
        Dataset of labels in string.
    filename : str
        Path to the hdf5 file.

    Returns
    -------
    None
    """
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "w") as hf:
        hf.create_dataset(
            "img", np.shape(img), h5py.h5t.IEEE_F32BE, compression="gzip", data=img
        )
        hf.create_dataset(
            "label", np.shape(label), compression="gzip", data=label.astype("S")
        )

def open_param_set_dir(i_path_param, out_dir):
    """
    Create a directory to save the results of the model with the parameter file used.
    The directory is named with the date and time of the execution.

    Parameters
    ----------
    i_path_param : str
        Path to the parameter file used.
    out_dir : str
        Path to the output directory.

    Returns
    -------
    str
        Path to the directory created.
    """
    now = datetime.now()
    saveto = os.path.join(out_dir, "R_" + now.strftime("%d%m%y_%HH%MM%S"))
    os.makedirs(saveto, exist_ok=True)
    shutil.copyfile(i_path_param, os.path.join(saveto, "param.yaml"))
    return saveto


def load_yaml(file_name):
    """
    Load a yaml file.

    Parameters
    ----------
    file_name : str
        Path to the yaml file.

    Returns
    -------
    dict
        Dictionary containing the parameters.
    """
    with open(file_name, "r") as f:
        opt = safe_load(f)
    return opt

def extract_zip(chemin_zip, chemin_extraction):
    """
    Extract a zip file.

    Parameters
    ----------
    chemin_zip : str
        Path to the zip file to extract.
    chemin_extraction : str
        Path to the directory where the zip file will be extracted.

    Returns
    -------
    int
        1 if the extraction is successful.
    """
    with zipfile.ZipFile(chemin_zip, "r") as fichier_zip:
        fichier_zip.extractall(chemin_extraction)
    return 1


def compresser_en_zip(chemin_dossier, chemin_zip):
    """
    Compress a directory in a zip file.

    Parameters
    ----------
    chemin_dossier : str
        Path to the directory to compress.
    chemin_zip : str
        Path to the zip file to create.

    Returns
    -------
    int
        1 if the compression is successful.
    """
    with zipfile.ZipFile(chemin_zip, "w", zipfile.ZIP_DEFLATED) as fichier_zip:
        for dossier_actuel, sous_dossiers, fichiers in os.walk(chemin_dossier):
            for fichier in fichiers:
                chemin_complet = os.path.join(dossier_actuel, fichier)
                chemin_rel = os.path.relpath(chemin_complet, chemin_dossier)
                fichier_zip.write(chemin_complet, chemin_rel)
    return 1

def set_folder(out_dir, args):
    """
    Create and organize folders for the study.

    Parameters
    ----------
    out_dir : str
        Path to the base output directory.
    args : dict
        Dictionary containing pipeline parameters.

    Returns
    -------
    str
        Path to the organized study folder.
    """
    pattern = re.compile(r"\[\['(.*?)_direct'\]")
    pipeline_names = []
    for item in args.pipeline:
        match = pattern.search(item)
        if match:
            pipeline_names.append(match.group(1) + "_direct")

    folders = ["results", "models"]

    for folder in folders:
        check_and_create_directory(os.path.join(out_dir, folder))

    for models_folder in pipeline_names:
        check_and_create_directory(os.path.join(out_dir, "models", models_folder))

    check_and_create_directory(os.path.join(out_dir, "results"))

    return os.path.join(out_dir, ""), pipeline_names


def check_and_create_directory(directory):
    """
    Check if a directory exists, and create it if it does not.

    Parameters
    ----------
    directory : str
        Path to the directory.
        
    Returns
    -------
    str
        Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_yaml(directory, name, data):

    with open(directory+"/"+name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)