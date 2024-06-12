import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd
import os
from utils.files_management import (
    check_and_create_directory
)
import matplotlib.pyplot as plt

import numpy as np


def plot_boxplots(metrics_dict, save_dir = None):
    """
    Create and save boxplots for each metric by model and by fold.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing various computed metrics for multiple models.
    save_dir : str, optional
        Directory where the plots will be saved. If None, plots are not saved.
    """
    save_dir = save_dir+"boxplot"
    check_and_create_directory(save_dir)
    flattened_metrics = []
    for model_name, metrics_list in metrics_dict.items():
        for fold_metrics in metrics_list:
            flattened_metrics.append({'model': model_name, **fold_metrics})
    
    metrics_df = pd.DataFrame(flattened_metrics)
    columns_to_drop = ['confusion_matrix', 'y_true', 'y_pred']
    columns_to_drop = [col for col in columns_to_drop if col in metrics_df.columns]
    metrics_df.drop(columns=columns_to_drop, inplace=True)

    
    # Create and save boxplots for each metric by model
    for column in metrics_df.columns:
        if column not in ['model', 'confusion_matrix', 'y_true', 'y_pred', 'fold']:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='model', y=column)
            plt.title(f'Boxplot of {column} by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'boxplot_{column}_by_model.png'))
            plt.close()

    # Create and save boxplots for each metric by fold
    for column in metrics_df.columns:
        if column not in ['model', 'confusion_matrix', 'y_true', 'y_pred', 'fold']:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='fold', y=column)
            plt.title(f'Boxplot of {column} by Fold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'boxplot_{column}_by_fold.png'))
            plt.close()



def plot_roc_curves(models_dict, save_dir=None):
    """
    Generate and save ROC curves for each model in the dictionary.

    Parameters
    ----------
    models_dict : dict
        A dictionary where keys are model names and values are lists of dictionaries containing
        'y_true' and 'y_pred' for each fold.
    save_dir : str, optional
        Directory where the plots will be saved. If None, plots will not be saved.

    Returns
    -------
    None
    """
    save_dir = save_dir+"roc_curve"
    check_and_create_directory(save_dir)

    for model_name, results in models_dict.items():
        try:
            plt.figure(figsize=(10, 8))

            mean_fpr = np.linspace(0, 1, 100)
            mean_tprs = []

            for fold_data in results:
                y_true = np.array(fold_data['y_true'])
                y_probas = np.array(fold_data['y_pred'])

                classes = np.unique(y_true)
                probas = y_probas

                for i in range(len(classes)):
                    fpr, tpr, _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
                    mean_tpr = np.interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                    mean_tprs.append(mean_tpr)

            mean_tpr = np.mean(mean_tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, label='Average ROC curve (area = {0:0.2f})'.format(mean_auc))

            for fold_data in results:
                y_true = np.array(fold_data['y_true'])
                y_probas = np.array(fold_data['y_pred'])

                classes = np.unique(y_true)
                probas = y_probas

                for i in range(len(classes)):
                    fpr, tpr, _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
                    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Class {classes[i]} fold {fold_data["fold"]} ROC curve')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize="medium")
            plt.ylabel('True Positive Rate', fontsize="medium")
            plt.title(f'Receiver Operating Characteristic for {model_name}', fontsize="medium")
            plt.legend(loc='lower right', fontsize="medium")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'roc_{model_name}.png'))
            plt.close()
        except Exception as e:
            print(f"An error occurred while plotting ROC curves for model {model_name}: {e}")
