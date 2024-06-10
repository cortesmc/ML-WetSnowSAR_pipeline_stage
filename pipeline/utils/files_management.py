import re, os, h5py, logging, pickle, shutil, zipfile
import numpy as np
import pandas as pd
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
    Open log file (.log) to parse it.

    Parameters
    ----------
    path_log : str
        Path to the log file.

    Returns
    -------
    list
        List of lines in the log file.
    """
    with open(path_log, "r") as f:
        log = f.readlines()
    return log


def clean_log(log):
    """
    Clean log file to remove useless information (line number, time, etc.) to make it more readable.

    Parameters
    ----------
    log : list
        List of lines in the log file.

    Returns
    -------
    list
        List of lines in the log file without useless information.
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


def report_metric_from_log(logg, dic):
    """
    Report various metrics from a dictionary containing the scores for each model in a log file.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance to log the report.
    dic : dict
        Dictionary containing various metrics for each model.

    Returns
    -------
    logging.Logger
        Logger instance with the reported metrics.
    """
    logg.info(f"================== Final report ==================")
    for model_name, model_metrics in dic.items():
        logg.info(f"__________________ Model : {model_name} __________________")
    
        metrics_to_report = ['f1_score_macro', 'f1_score_weighted', 'accuracy_score', 
                             'precision_score_macro', 'recall_score_macro', 'roc_auc_score', 
                             'log_loss', 'kappa_score', 'confusion_matrix']

        for metric in metrics_to_report:
            try:
                if metric in model_metrics[0]:
                    values = [fold_metrics[metric] for fold_metrics in model_metrics]

                if isinstance(values[0], (int, float)):
                    logg.info(f"{metric} : {np.mean(values)} +/- {np.std(values)}")
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
    Compute various classification metrics and report them in a log file.
    The y_true and y_pred must be categorical (one hot encoded) or binary (0 or 1).

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    le : LabelEncoder
        LabelEncoder object.
    logg : logging.Logger
        Logger instance.
    fold : int
        Fold number.

    Returns
    -------
    logging.Logger
        Logger instance.
    dict
        Dictionary containing various computed metrics.
    """
    
    if y_pred.shape[1] > 1:
        y_pred_classes = y_pred.argmax(axis=1)
    elif y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1)
    else:
        y_true = y_true.ravel()
        y_pred_classes = y_pred.ravel()
        y_pred_classes = np.argmax(y_pred_classes, axis=1)
        
    y_true_transformed = le.inverse_transform(y_true)
    y_pred_transformed = le.inverse_transform(y_pred_classes)
   
    all_labels = le.classes_

    cm = confusion_matrix(y_true_transformed, y_pred_transformed, labels=all_labels)
    cm_df = pd.DataFrame(
        100 * cm.astype(float) / cm.sum(axis=1, keepdims=True),
        columns=all_labels,
        index=all_labels
    ).round(4).fillna(0)
    
    f1_macro = 100 * round(f1_score(y_true, y_pred_classes, average="macro"), 4)
    f1_weighted = 100 * round(f1_score(y_true, y_pred_classes, average="weighted"), 4)
    accuracy = 100 * round(accuracy_score(y_true, y_pred_classes), 4)
    precision_macro = 100 * round(precision_score(y_true, y_pred_classes, average="macro"), 4)
    recall_macro = 100 * round(recall_score(y_true, y_pred_classes, average="macro"), 4)
    roc_auc = 100 * round(roc_auc_score(y_true, y_pred_classes, multi_class="ovr"), 4)
    log_loss_val = 100 * round(log_loss(y_true, y_pred), 4)
    kappa = 100 * round(cohen_kappa_score(y_true, y_pred_classes), 4)
    
    metrics = {
        'f1_score_macro': f1_macro,
        'f1_score_weighted': f1_weighted,
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
    
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['fold'] = fold
    return logg, metrics


def init_logger(path_log, name ):
    """
    Initialize a logger.

    Parameters
    ----------
    path_log : str
        Path to the log file.
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Logger instance.
    str
        Path to the log file.
    """
    datestr = "%m/%d/%Y-%I:%M:%S %p"
    filename = os.path.join(path_log, f"log_{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s", datefmt=datestr)
    fh.setFormatter(formatter)

    if not logger.handlers:  
        logger.addHandler(fh)

    logger.info("Started")
    return logger, filename

def logger_dataset(logg, x, metadata, targets, pipeline_param):
    """
    Log information about the dataset.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance.
    x : numpy.ndarray
        Input data.
    metadata : dict
        Metadata dictionary.
    targets : numpy.ndarray
        Target labels.
    pipeline_param : dict
        Pipeline parameters.

    Returns
    -------
    logging.Logger
        Logger instance.
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

def logger_fold(logg, fold_groupes, targets, metadata):
    """
    Log information about each fold in cross-validation.

    Parameters
    ----------
    logg : logging.Logger
        Logger instance.
    fold_groupes : list of tuples
        List of (train_index, test_index) tuples for each fold.
    targets : numpy.ndarray
        Target labels.
    metadata : dict
        Metadata dictionary.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    for kfold, (train_index, test_index) in enumerate(fold_groupes):
        logg.info(f"------------------ Fold: {kfold} ------------------")
        
        train_unique_targets, train_target_counts = np.unique(targets[train_index], return_counts=True)
        train_target_ratios = train_target_counts / train_target_counts.sum()
        train_target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(train_unique_targets, train_target_counts, train_target_ratios))
        
        test_unique_targets, test_target_counts = np.unique(targets[test_index], return_counts=True)
        test_target_ratios = test_target_counts / test_target_counts.sum()
        test_target_info = ", ".join(f"{target}: {count} ({ratio:.2%})" for target, count, ratio in zip(test_unique_targets, test_target_counts, test_target_ratios))
        
        logg.info(f"    - Distribution class train: {train_target_info}")
        logg.info(f"    - Distribution class test: {test_target_info}")
        logg.info(f"    - Train size: {len(train_index) / (len(train_index) + len(test_index)) * 100:.2f}%")
        logg.info(f"    - Massif in train: {np.unique(metadata['metadata'][train_index, 1])}")
        logg.info(f"    - Massif in test: {np.unique(metadata['metadata'][test_index, 1])}")

    return logg


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
        )  # IEEE_F32BE is big endian float32
        hf.create_dataset(
            "label", np.shape(label), compression="gzip", data=label.astype("S")
        )


def load_h5(filename):
    """
    Load image and label from a hdf5 file.

    Parameters
    ----------
    filename : str
        Path to the hdf5 file.

    Returns
    -------
    numpy.ndarray
        Dataset of images in float32.
    numpy.ndarray
        Dataset of labels in string.
    """
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "r") as hf:
        data = np.array(hf["img"][:]).astype(np.float32)
        meta = np.array(hf["label"][:]).astype(str)
    return data, meta


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

def set_folder(out_dir, pipeline_param):
    """
    Create and organize folders for the study.

    Parameters
    ----------
    out_dir : str
        Path to the base output directory.
    pipeline_param : dict
        Dictionary containing pipeline parameters.

    Returns
    -------
    str
        Path to the organized study folder.
    """
    folders= ["results", "models", "html"]
    now = datetime.now()
    date = now.strftime("%d%m%y_%HH%MM%S")
    folder_name = f"study_{date}_{pipeline_param['labeling_method']}_{pipeline_param['fold_method']}"
    out_dir = check_and_create_directory(out_dir+folder_name)
    for folder in folders:
        check_and_create_directory(out_dir+f"/{folder}")

    models= pipeline_param["pipeline_names"]
    for models_folder in models:
        check_and_create_directory(out_dir+f"/models/{models_folder}")

    check_and_create_directory(out_dir+f"/results")
    return out_dir+"/"


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
