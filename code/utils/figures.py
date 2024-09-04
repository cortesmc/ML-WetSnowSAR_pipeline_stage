import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd
import os
from utils.files_management import check_and_create_directory

def plot_boxplots(metrics_dict, metrics_to_plot=[], save_dir=None, fold_key={}, labels_massives=False):
    """
    Create and save boxplots for specified metrics by model and by fold.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing various computed metrics for multiple models.
    
    metrics_to_plot : list, optional
        List of metrics to be plotted. If empty, all metrics are plotted.
    
    save_dir : str, optional
        Directory where the plots will be saved. If None, plots are not saved.
    
    fold_key : dict, optional
        Mapping of fold identifiers to more descriptive names, used if `labels_massives` is True.
    
    labels_massives : bool, optional
        If True, uses descriptive names from `fold_key` for the x-axis labels in fold plots.
    
    Returns
    -------
    None
    """
    save_dir = os.path.join(save_dir, "boxplot") if save_dir else None
    check_and_create_directory(save_dir)

    flattened_metrics = []
    for model_name, metrics_list in metrics_dict.items():
        for fold_metrics in metrics_list:
            flattened_metrics.append({'model': model_name, **fold_metrics})

    metrics_df = pd.DataFrame(flattened_metrics)
    columns_to_drop = ['confusion_matrix', 'y_true', 'y_pred', 'f1_score_multiclass']
    metrics_df.drop(columns=[col for col in columns_to_drop if col in metrics_df.columns], inplace=True)

    # Create and save boxplots for each metric by model
    for column in metrics_df.columns:
        if column in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='model', y=column)
            plt.title(f'Boxplot of {column} by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'boxplot_{column}_by_model.png'))
            plt.close()

    if labels_massives:
        metrics_df['fold'] = metrics_df['fold'].map(lambda x: fold_key[x]['test'][0])

    # Create and save boxplots for each metric by fold
    for column in metrics_df.columns:
        if column in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='fold', y=column)
            plt.title(f'Boxplot of {column} by {"Massif" if labels_massives else "Fold"}')
            if labels_massives:
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'boxplot_{column}_by_fold.png'))
            plt.close()


def plot_roc_curves(models_dict, save_dir=None):
    """
    Generate and save ROC curves for each model in the provided dictionary.

    Parameters
    ----------
    models_dict : dict
        A dictionary where keys are model names and values are lists of dictionaries
        containing 'y_true' and 'y_pred' for each fold.
    
    save_dir : str, optional
        Directory where the plots will be saved. If None, plots will not be saved.
    
    Returns
    -------
    None
    """
    save_dir = os.path.join(save_dir, "roc_curve") if save_dir else None
    check_and_create_directory(save_dir)

    for model_name, results in models_dict.items():
        try:
            plt.figure(figsize=(10, 8))
            mean_fpr = np.linspace(0, 1, 100)
            mean_tprs = {}
            mean_aucs = {}
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
            color_dict = {}

            for fold_data in results:
                y_true = np.array(fold_data['y_true'])
                y_probas = np.array(fold_data['y_pred'])
                classes = np.unique(y_true)

                for i, class_i in enumerate(classes):
                    if class_i not in color_dict:
                        color_dict[class_i] = colors.pop(0)

                    fpr, tpr, _ = roc_curve(y_true, y_probas[:, i], pos_label=class_i)
                    mean_tpr = np.interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0

                    if class_i not in mean_tprs:
                        mean_tprs[class_i] = []
                    mean_tprs[class_i].append(mean_tpr)

                    plt.plot(fpr, tpr, lw=1, alpha=0.3, color=color_dict[class_i])

            for class_i in mean_tprs.keys():
                mean_tpr = np.mean(mean_tprs[class_i], axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                mean_aucs[class_i] = mean_auc

                plt.plot(mean_fpr, mean_tpr, lw=2, color=color_dict[class_i],
                         label=f'Class {class_i} mean ROC (area = {mean_auc:0.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize="medium")
            plt.ylabel('True Positive Rate', fontsize="medium")
            plt.title(f'Receiver Operating Characteristic for {model_name}', fontsize="medium")
            plt.legend(loc='lower right', fontsize="medium")
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'roc_{model_name}.png'))
            plt.close()
        except Exception as e:
            print(f"An error occurred while plotting ROC curves for model {model_name}: {e}")
