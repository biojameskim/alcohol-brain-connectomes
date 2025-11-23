"""
Name: logreg_cross_sex_repeated.py
Purpose: Perform cross-sex testing using fold-based approach
      - For each repeat (100) and each outer fold (5):
        - Train on 4/5 males → test on ALL females (500 tests total)
        - Train on 4/5 females → test on ALL males (500 tests total)
      - Uses nested CV for hyperparameter selection (LogisticRegressionCV)
      - Generates 500 data points per direction for violin plots
      - Mirrors the approach from logreg_master.py but tests on opposite sex
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.pipeline import Pipeline


def save_results(models, metrics_male_to_female, metrics_female_to_male,
                 ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
                 simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male,
                 control_only):
  """
  [save_results] saves the resulting metrics from cross-sex testing to NumPy files.
  """
  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  # Male to Female results
  for model_type in models:
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_accuracies.npy',
            metrics_male_to_female[model_type]['accuracies'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_balanced_accuracies.npy',
            metrics_male_to_female[model_type]['balanced_accuracies'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_roc_aucs.npy',
            metrics_male_to_female[model_type]['roc_aucs'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_pr_aucs.npy',
            metrics_male_to_female[model_type]['pr_aucs'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_coefficients.npy',
            metrics_male_to_female[model_type]['coefficients'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_true_labels.npy',
            metrics_male_to_female[model_type]['all_true_labels'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_male_to_female_pred_probs.npy',
            metrics_male_to_female[model_type]['all_pred_probs'])

  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_accuracies.npy',
          ensemble_metrics_male_to_female['accuracies'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_balanced_accuracies.npy',
          ensemble_metrics_male_to_female['balanced_accuracies'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_roc_aucs.npy',
          ensemble_metrics_male_to_female['roc_aucs'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_pr_aucs.npy',
          ensemble_metrics_male_to_female['pr_aucs'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_coefficients.npy',
          ensemble_metrics_male_to_female['coefficients'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_true_labels.npy',
          ensemble_metrics_male_to_female['all_true_labels'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_pred_probs.npy',
          ensemble_metrics_male_to_female['all_pred_probs'])

  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_accuracies.npy',
          simple_ensemble_metrics_male_to_female['accuracies'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_balanced_accuracies.npy',
          simple_ensemble_metrics_male_to_female['balanced_accuracies'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_roc_aucs.npy',
          simple_ensemble_metrics_male_to_female['roc_aucs'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_pr_aucs.npy',
          simple_ensemble_metrics_male_to_female['pr_aucs'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_true_labels.npy',
          simple_ensemble_metrics_male_to_female['all_true_labels'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_male_to_female_pred_probs.npy',
          simple_ensemble_metrics_male_to_female['all_pred_probs'])

  # Female to Male results
  for model_type in models:
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_accuracies.npy',
            metrics_female_to_male[model_type]['accuracies'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_balanced_accuracies.npy',
            metrics_female_to_male[model_type]['balanced_accuracies'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_roc_aucs.npy',
            metrics_female_to_male[model_type]['roc_aucs'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_pr_aucs.npy',
            metrics_female_to_male[model_type]['pr_aucs'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_coefficients.npy',
            metrics_female_to_male[model_type]['coefficients'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_true_labels.npy',
            metrics_female_to_male[model_type]['all_true_labels'])
    np.save(f'results/{model_type}/logreg_{model_type}_{file_name}_female_to_male_pred_probs.npy',
            metrics_female_to_male[model_type]['all_pred_probs'])

  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_accuracies.npy',
          ensemble_metrics_female_to_male['accuracies'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_balanced_accuracies.npy',
          ensemble_metrics_female_to_male['balanced_accuracies'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_roc_aucs.npy',
          ensemble_metrics_female_to_male['roc_aucs'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_pr_aucs.npy',
          ensemble_metrics_female_to_male['pr_aucs'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_coefficients.npy',
          ensemble_metrics_female_to_male['coefficients'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_true_labels.npy',
          ensemble_metrics_female_to_male['all_true_labels'])
  np.save(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_pred_probs.npy',
          ensemble_metrics_female_to_male['all_pred_probs'])

  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_accuracies.npy',
          simple_ensemble_metrics_female_to_male['accuracies'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_balanced_accuracies.npy',
          simple_ensemble_metrics_female_to_male['balanced_accuracies'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_roc_aucs.npy',
          simple_ensemble_metrics_female_to_male['roc_aucs'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_pr_aucs.npy',
          simple_ensemble_metrics_female_to_male['pr_aucs'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_true_labels.npy',
          simple_ensemble_metrics_female_to_male['all_true_labels'])
  np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}_female_to_male_pred_probs.npy',
          simple_ensemble_metrics_female_to_male['all_pred_probs'])


def create_metrics_report(metrics_male_to_female, metrics_female_to_male,
                         ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
                         simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male,
                         num_splits, num_repeats, control_only, save_to_file=False):
  """
  [create_metrics_report] creates a report of the cross-sex testing metrics
  """
  if control_only:
    control_str = "Baseline subjects with cahalan=='control' only"
  else:
    control_str = "Baseline subjects with cahalan=='control' or cahalan=='moderate'"

  num_tests = num_splits * num_repeats

  report_lines = [
    "Results for cross-sex testing (Fold-based approach):\n",
    f"Number of Repeats: {num_repeats}",
    f"Number of Folds per Repeat: {num_splits}",
    f"Total number of tests: {num_tests}",
    f"{control_str}\n",

    "=" * 50,
    "MALE TO FEMALE (Train on males, test on females)",
    "=" * 50,
    "",
    "Individual Models:",
    "SC:",
    f"  Mean accuracy: {np.mean(metrics_male_to_female['SC']['accuracies']):.4f} ± {np.std(metrics_male_to_female['SC']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_male_to_female['SC']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_male_to_female['SC']['balanced_accuracies']):.4f} ± {np.std(metrics_male_to_female['SC']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_male_to_female['SC']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_male_to_female['SC']['roc_aucs']):.4f} ± {np.std(metrics_male_to_female['SC']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_male_to_female['SC']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_male_to_female['SC']['pr_aucs']):.4f} ± {np.std(metrics_male_to_female['SC']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_male_to_female['SC']['pr_aucs']):.4f}",
    "",
    "FC:",
    f"  Mean accuracy: {np.mean(metrics_male_to_female['FC']['accuracies']):.4f} ± {np.std(metrics_male_to_female['FC']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_male_to_female['FC']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_male_to_female['FC']['balanced_accuracies']):.4f} ± {np.std(metrics_male_to_female['FC']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_male_to_female['FC']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_male_to_female['FC']['roc_aucs']):.4f} ± {np.std(metrics_male_to_female['FC']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_male_to_female['FC']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_male_to_female['FC']['pr_aucs']):.4f} ± {np.std(metrics_male_to_female['FC']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_male_to_female['FC']['pr_aucs']):.4f}",
    "",
    "Demographics:",
    f"  Mean accuracy: {np.mean(metrics_male_to_female['demos']['accuracies']):.4f} ± {np.std(metrics_male_to_female['demos']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_male_to_female['demos']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_male_to_female['demos']['balanced_accuracies']):.4f} ± {np.std(metrics_male_to_female['demos']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_male_to_female['demos']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_male_to_female['demos']['roc_aucs']):.4f} ± {np.std(metrics_male_to_female['demos']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_male_to_female['demos']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_male_to_female['demos']['pr_aucs']):.4f} ± {np.std(metrics_male_to_female['demos']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_male_to_female['demos']['pr_aucs']):.4f}",
    "",
    "Ensemble Models:",
    "Ensemble:",
    f"  Mean accuracy: {np.mean(ensemble_metrics_male_to_female['accuracies']):.4f} ± {np.std(ensemble_metrics_male_to_female['accuracies']):.4f}",
    f"  Median accuracy: {np.median(ensemble_metrics_male_to_female['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(ensemble_metrics_male_to_female['balanced_accuracies']):.4f} ± {np.std(ensemble_metrics_male_to_female['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(ensemble_metrics_male_to_female['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(ensemble_metrics_male_to_female['roc_aucs']):.4f} ± {np.std(ensemble_metrics_male_to_female['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(ensemble_metrics_male_to_female['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(ensemble_metrics_male_to_female['pr_aucs']):.4f} ± {np.std(ensemble_metrics_male_to_female['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(ensemble_metrics_male_to_female['pr_aucs']):.4f}",
    "",
    "Simple Ensemble:",
    f"  Mean accuracy: {np.mean(simple_ensemble_metrics_male_to_female['accuracies']):.4f} ± {np.std(simple_ensemble_metrics_male_to_female['accuracies']):.4f}",
    f"  Median accuracy: {np.median(simple_ensemble_metrics_male_to_female['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(simple_ensemble_metrics_male_to_female['balanced_accuracies']):.4f} ± {np.std(simple_ensemble_metrics_male_to_female['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(simple_ensemble_metrics_male_to_female['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(simple_ensemble_metrics_male_to_female['roc_aucs']):.4f} ± {np.std(simple_ensemble_metrics_male_to_female['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(simple_ensemble_metrics_male_to_female['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(simple_ensemble_metrics_male_to_female['pr_aucs']):.4f} ± {np.std(simple_ensemble_metrics_male_to_female['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(simple_ensemble_metrics_male_to_female['pr_aucs']):.4f}",
    "",
    "=" * 50,
    "FEMALE TO MALE (Train on females, test on males)",
    "=" * 50,
    "",
    "Individual Models:",
    "SC:",
    f"  Mean accuracy: {np.mean(metrics_female_to_male['SC']['accuracies']):.4f} ± {np.std(metrics_female_to_male['SC']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_female_to_male['SC']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_female_to_male['SC']['balanced_accuracies']):.4f} ± {np.std(metrics_female_to_male['SC']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_female_to_male['SC']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_female_to_male['SC']['roc_aucs']):.4f} ± {np.std(metrics_female_to_male['SC']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_female_to_male['SC']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_female_to_male['SC']['pr_aucs']):.4f} ± {np.std(metrics_female_to_male['SC']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_female_to_male['SC']['pr_aucs']):.4f}",
    "",
    "FC:",
    f"  Mean accuracy: {np.mean(metrics_female_to_male['FC']['accuracies']):.4f} ± {np.std(metrics_female_to_male['FC']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_female_to_male['FC']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_female_to_male['FC']['balanced_accuracies']):.4f} ± {np.std(metrics_female_to_male['FC']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_female_to_male['FC']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_female_to_male['FC']['roc_aucs']):.4f} ± {np.std(metrics_female_to_male['FC']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_female_to_male['FC']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_female_to_male['FC']['pr_aucs']):.4f} ± {np.std(metrics_female_to_male['FC']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_female_to_male['FC']['pr_aucs']):.4f}",
    "",
    "Demographics:",
    f"  Mean accuracy: {np.mean(metrics_female_to_male['demos']['accuracies']):.4f} ± {np.std(metrics_female_to_male['demos']['accuracies']):.4f}",
    f"  Median accuracy: {np.median(metrics_female_to_male['demos']['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(metrics_female_to_male['demos']['balanced_accuracies']):.4f} ± {np.std(metrics_female_to_male['demos']['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(metrics_female_to_male['demos']['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(metrics_female_to_male['demos']['roc_aucs']):.4f} ± {np.std(metrics_female_to_male['demos']['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(metrics_female_to_male['demos']['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(metrics_female_to_male['demos']['pr_aucs']):.4f} ± {np.std(metrics_female_to_male['demos']['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(metrics_female_to_male['demos']['pr_aucs']):.4f}",
    "",
    "Ensemble Models:",
    "Ensemble:",
    f"  Mean accuracy: {np.mean(ensemble_metrics_female_to_male['accuracies']):.4f} ± {np.std(ensemble_metrics_female_to_male['accuracies']):.4f}",
    f"  Median accuracy: {np.median(ensemble_metrics_female_to_male['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(ensemble_metrics_female_to_male['balanced_accuracies']):.4f} ± {np.std(ensemble_metrics_female_to_male['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(ensemble_metrics_female_to_male['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(ensemble_metrics_female_to_male['roc_aucs']):.4f} ± {np.std(ensemble_metrics_female_to_male['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(ensemble_metrics_female_to_male['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(ensemble_metrics_female_to_male['pr_aucs']):.4f} ± {np.std(ensemble_metrics_female_to_male['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(ensemble_metrics_female_to_male['pr_aucs']):.4f}",
    "",
    "Simple Ensemble:",
    f"  Mean accuracy: {np.mean(simple_ensemble_metrics_female_to_male['accuracies']):.4f} ± {np.std(simple_ensemble_metrics_female_to_male['accuracies']):.4f}",
    f"  Median accuracy: {np.median(simple_ensemble_metrics_female_to_male['accuracies']):.4f}",
    f"  Mean balanced accuracy: {np.mean(simple_ensemble_metrics_female_to_male['balanced_accuracies']):.4f} ± {np.std(simple_ensemble_metrics_female_to_male['balanced_accuracies']):.4f}",
    f"  Median balanced accuracy: {np.median(simple_ensemble_metrics_female_to_male['balanced_accuracies']):.4f}",
    f"  Mean ROC AUC: {np.mean(simple_ensemble_metrics_female_to_male['roc_aucs']):.4f} ± {np.std(simple_ensemble_metrics_female_to_male['roc_aucs']):.4f}",
    f"  Median ROC AUC: {np.median(simple_ensemble_metrics_female_to_male['roc_aucs']):.4f}",
    f"  Mean PR AUC: {np.mean(simple_ensemble_metrics_female_to_male['pr_aucs']):.4f} ± {np.std(simple_ensemble_metrics_female_to_male['pr_aucs']):.4f}",
    f"  Median PR AUC: {np.median(simple_ensemble_metrics_female_to_male['pr_aucs']):.4f}",
  ]

  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  if save_to_file:
    with open(f'results/reports/logreg_metrics/logreg_cross_sex_repeated_{file_name}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  else:
    print("\n".join(report_lines))


def cross_sex_testing_repeated(X_dict_male, y_male, site_male, X_dict_female, y_female, site_female,
                                n_splits, num_repeats, random_ints):
  """
  Performs cross-sex testing using fold-based approach (mirrors logreg_master.py)
  For each repeat and each fold:
    - Train on 4/5 of males → test on ALL females
    - Train on 4/5 of females → test on ALL males
  This generates 500 test results per direction (100 repeats × 5 folds)
  """
  C_values = np.logspace(-4, 4, 15)

  # Initialize metrics dictionaries for both directions
  # Now storing num_repeats values (100 total) - averaging across folds like logreg_master.py
  metrics_male_to_female = {}
  metrics_female_to_male = {}

  for model in MODELS:
    metrics_male_to_female[model] = {
        "accuracies": np.empty(num_repeats),
        "balanced_accuracies": np.empty(num_repeats),
        "roc_aucs": np.empty(num_repeats),
        "pr_aucs": np.empty(num_repeats),
        "coefficients": np.empty((num_repeats * n_splits, X_dict_male[model].shape[1])),
        "all_true_labels": [],
        "all_pred_probs": []
    }
    metrics_female_to_male[model] = {
        "accuracies": np.empty(num_repeats),
        "balanced_accuracies": np.empty(num_repeats),
        "roc_aucs": np.empty(num_repeats),
        "pr_aucs": np.empty(num_repeats),
        "coefficients": np.empty((num_repeats * n_splits, X_dict_female[model].shape[1])),
        "all_true_labels": [],
        "all_pred_probs": []
    }

  ensemble_metrics_male_to_female = {
      "accuracies": np.empty(num_repeats),
      "balanced_accuracies": np.empty(num_repeats),
      "roc_aucs": np.empty(num_repeats),
      "pr_aucs": np.empty(num_repeats),
      "coefficients": np.empty((num_repeats * n_splits, len(MODELS))),
      "all_true_labels": [],
      "all_pred_probs": []
  }
  ensemble_metrics_female_to_male = {
      "accuracies": np.empty(num_repeats),
      "balanced_accuracies": np.empty(num_repeats),
      "roc_aucs": np.empty(num_repeats),
      "pr_aucs": np.empty(num_repeats),
      "coefficients": np.empty((num_repeats * n_splits, len(MODELS))),
      "all_true_labels": [],
      "all_pred_probs": []
  }

  simple_ensemble_metrics_male_to_female = {
      "accuracies": np.empty(num_repeats),
      "balanced_accuracies": np.empty(num_repeats),
      "roc_aucs": np.empty(num_repeats),
      "pr_aucs": np.empty(num_repeats),
      "all_true_labels": [],
      "all_pred_probs": []
  }
  simple_ensemble_metrics_female_to_male = {
      "accuracies": np.empty(num_repeats),
      "balanced_accuracies": np.empty(num_repeats),
      "roc_aucs": np.empty(num_repeats),
      "pr_aucs": np.empty(num_repeats),
      "all_true_labels": [],
      "all_pred_probs": []
  }

  print(f"Starting cross-sex testing: {num_repeats} repeats × {n_splits} folds")
  print(f"Training {len(MODELS)} models + 2 ensemble models per iteration\n")
  repeat_start_time = time.time()

  for repeat_idx in range(num_repeats):
    random_state = random_ints[repeat_idx]

    # Create outer fold splits for males and females (like logreg_master.py)
    outer_kf_male = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_kf_female = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Create stratification keys
    n_samples_male = len(y_male)
    n_samples_female = len(y_female)
    simple_site_male = np.argmax(site_male, axis=1)
    simple_site_female = np.argmax(site_female, axis=1)
    stratification_key_male = [str(a) + '_' + str(b) for a, b in zip(y_male, simple_site_male)]
    stratification_key_female = [str(a) + '_' + str(b) for a, b in zip(y_female, simple_site_female)]

    # Initialize outer loop metrics for this repeat (to average across folds)
    outer_loop_metrics_male_to_female = {}
    for model_type in MODELS:
      outer_loop_metrics_male_to_female[model_type] = {
          "accuracies": np.empty(n_splits),
          "balanced_accuracies": np.empty(n_splits),
          "roc_aucs": np.empty(n_splits),
          "pr_aucs": np.empty(n_splits),
          "coefs": np.empty((n_splits, X_dict_male[model_type].shape[1]))
      }

    ensemble_outer_loop_metrics_male_to_female = {
        "accuracies": np.empty(n_splits),
        "balanced_accuracies": np.empty(n_splits),
        "roc_aucs": np.empty(n_splits),
        "pr_aucs": np.empty(n_splits),
        "coefs": np.empty((n_splits, len(MODELS)))
    }

    simple_ensemble_outer_loop_metrics_male_to_female = {
        "accuracies": np.empty(n_splits),
        "balanced_accuracies": np.empty(n_splits),
        "roc_aucs": np.empty(n_splits),
        "pr_aucs": np.empty(n_splits)
    }

    # ===== MALE TO FEMALE =====
    # For each fold, train on male train set and test on ALL females
    for fold_idx, (train_index_male, test_index_male) in enumerate(outer_kf_male.split(np.zeros(n_samples_male), stratification_key_male)):
      test_idx = repeat_idx * n_splits + fold_idx
      base_models_male = []

      for model_type in MODELS:
        X_train = X_dict_male[model_type][train_index_male]
        X_test = X_dict_female[model_type]  # Test on ALL females
        y_train = y_male[train_index_male]
        y_test = y_female  # ALL female labels

        inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold_idx)

        # Create pipeline with standardization and logistic regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        base_models_male.append(pipeline)

        # Get predictions on all females
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate and store metrics for this fold
        outer_loop_metrics_male_to_female[model_type]['accuracies'][fold_idx] = accuracy_score(y_test, y_pred)
        outer_loop_metrics_male_to_female[model_type]['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_test, y_pred)
        outer_loop_metrics_male_to_female[model_type]['roc_aucs'][fold_idx] = roc_auc_score(y_test, y_prob)
        outer_loop_metrics_male_to_female[model_type]['pr_aucs'][fold_idx] = average_precision_score(y_test, y_prob)
        outer_loop_metrics_male_to_female[model_type]['coefs'][fold_idx] = pipeline.named_steps['classifier'].coef_[0]
        metrics_male_to_female[model_type]['all_true_labels'].extend(y_test)
        metrics_male_to_female[model_type]['all_pred_probs'].extend(y_prob)

      # Ensemble model (male to female)
      inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold_idx)
      train_preds = np.column_stack([
          cross_val_predict(
              Pipeline([
                  ('scaler', StandardScaler()),
                  ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
              ]),
              X_dict_male[model_type][train_index_male], y_male[train_index_male],
              method='predict_proba', cv=inner_kf, n_jobs=-1
          )[:, 1]
          for model_type in MODELS
      ])

      test_preds = np.column_stack([
          base_models_male[i].predict_proba(X_dict_female[model_type])[:, 1]
          for i, model_type in enumerate(MODELS)
      ])

      ensemble_model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
      ensemble_model.fit(train_preds, y_male[train_index_male])

      y_pred = ensemble_model.predict(test_preds)
      y_prob = ensemble_model.predict_proba(test_preds)[:, 1]

      ensemble_outer_loop_metrics_male_to_female['accuracies'][fold_idx] = accuracy_score(y_female, y_pred)
      ensemble_outer_loop_metrics_male_to_female['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_female, y_pred)
      ensemble_outer_loop_metrics_male_to_female['roc_aucs'][fold_idx] = roc_auc_score(y_female, y_prob)
      ensemble_outer_loop_metrics_male_to_female['pr_aucs'][fold_idx] = average_precision_score(y_female, y_prob)
      ensemble_outer_loop_metrics_male_to_female['coefs'][fold_idx] = ensemble_model.coef_[0]
      ensemble_metrics_male_to_female['all_true_labels'].extend(y_female)
      ensemble_metrics_male_to_female['all_pred_probs'].extend(y_prob)

      # Simple ensemble (male to female)
      averaged_preds = np.mean([
          base_models_male[i].predict_proba(X_dict_female[model_type])[:, 1]
          for i, model_type in enumerate(MODELS)
      ], axis=0)
      y_pred = (averaged_preds >= 0.5).astype(int)
      y_prob = averaged_preds

      simple_ensemble_outer_loop_metrics_male_to_female['accuracies'][fold_idx] = accuracy_score(y_female, y_pred)
      simple_ensemble_outer_loop_metrics_male_to_female['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_female, y_pred)
      simple_ensemble_outer_loop_metrics_male_to_female['roc_aucs'][fold_idx] = roc_auc_score(y_female, y_prob)
      simple_ensemble_outer_loop_metrics_male_to_female['pr_aucs'][fold_idx] = average_precision_score(y_female, y_prob)
      simple_ensemble_metrics_male_to_female['all_true_labels'].extend(y_female)
      simple_ensemble_metrics_male_to_female['all_pred_probs'].extend(y_prob)

    # Initialize outer loop metrics for female to male
    outer_loop_metrics_female_to_male = {}
    for model_type in MODELS:
      outer_loop_metrics_female_to_male[model_type] = {
          "accuracies": np.empty(n_splits),
          "balanced_accuracies": np.empty(n_splits),
          "roc_aucs": np.empty(n_splits),
          "pr_aucs": np.empty(n_splits),
          "coefs": np.empty((n_splits, X_dict_female[model_type].shape[1]))
      }

    ensemble_outer_loop_metrics_female_to_male = {
        "accuracies": np.empty(n_splits),
        "balanced_accuracies": np.empty(n_splits),
        "roc_aucs": np.empty(n_splits),
        "pr_aucs": np.empty(n_splits),
        "coefs": np.empty((n_splits, len(MODELS)))
    }

    simple_ensemble_outer_loop_metrics_female_to_male = {
        "accuracies": np.empty(n_splits),
        "balanced_accuracies": np.empty(n_splits),
        "roc_aucs": np.empty(n_splits),
        "pr_aucs": np.empty(n_splits)
    }

    # ===== FEMALE TO MALE =====
    # For each fold, train on female train set and test on ALL males
    for fold_idx, (train_index_female, test_index_female) in enumerate(outer_kf_female.split(np.zeros(n_samples_female), stratification_key_female)):
      test_idx = repeat_idx * n_splits + fold_idx
      base_models_female = []

      for model_type in MODELS:
        X_train = X_dict_female[model_type][train_index_female]
        X_test = X_dict_male[model_type]  # Test on ALL males
        y_train = y_female[train_index_female]
        y_test = y_male  # ALL male labels

        inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold_idx)

        # Create pipeline with standardization and logistic regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        base_models_female.append(pipeline)

        # Get predictions on all males
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate and store metrics for this fold
        outer_loop_metrics_female_to_male[model_type]['accuracies'][fold_idx] = accuracy_score(y_test, y_pred)
        outer_loop_metrics_female_to_male[model_type]['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_test, y_pred)
        outer_loop_metrics_female_to_male[model_type]['roc_aucs'][fold_idx] = roc_auc_score(y_test, y_prob)
        outer_loop_metrics_female_to_male[model_type]['pr_aucs'][fold_idx] = average_precision_score(y_test, y_prob)
        outer_loop_metrics_female_to_male[model_type]['coefs'][fold_idx] = pipeline.named_steps['classifier'].coef_[0]
        metrics_female_to_male[model_type]['all_true_labels'].extend(y_test)
        metrics_female_to_male[model_type]['all_pred_probs'].extend(y_prob)

      # Ensemble model (female to male)
      inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold_idx)
      train_preds = np.column_stack([
          cross_val_predict(
              Pipeline([
                  ('scaler', StandardScaler()),
                  ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
              ]),
              X_dict_female[model_type][train_index_female], y_female[train_index_female],
              method='predict_proba', cv=inner_kf, n_jobs=-1
          )[:, 1]
          for model_type in MODELS
      ])

      test_preds = np.column_stack([
          base_models_female[i].predict_proba(X_dict_male[model_type])[:, 1]
          for i, model_type in enumerate(MODELS)
      ])

      ensemble_model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
      ensemble_model.fit(train_preds, y_female[train_index_female])

      y_pred = ensemble_model.predict(test_preds)
      y_prob = ensemble_model.predict_proba(test_preds)[:, 1]

      ensemble_outer_loop_metrics_female_to_male['accuracies'][fold_idx] = accuracy_score(y_male, y_pred)
      ensemble_outer_loop_metrics_female_to_male['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_male, y_pred)
      ensemble_outer_loop_metrics_female_to_male['roc_aucs'][fold_idx] = roc_auc_score(y_male, y_prob)
      ensemble_outer_loop_metrics_female_to_male['pr_aucs'][fold_idx] = average_precision_score(y_male, y_prob)
      ensemble_outer_loop_metrics_female_to_male['coefs'][fold_idx] = ensemble_model.coef_[0]
      ensemble_metrics_female_to_male['all_true_labels'].extend(y_male)
      ensemble_metrics_female_to_male['all_pred_probs'].extend(y_prob)

      # Simple ensemble (female to male)
      averaged_preds = np.mean([
          base_models_female[i].predict_proba(X_dict_male[model_type])[:, 1]
          for i, model_type in enumerate(MODELS)
      ], axis=0)
      y_pred = (averaged_preds >= 0.5).astype(int)
      y_prob = averaged_preds

      simple_ensemble_outer_loop_metrics_female_to_male['accuracies'][fold_idx] = accuracy_score(y_male, y_pred)
      simple_ensemble_outer_loop_metrics_female_to_male['balanced_accuracies'][fold_idx] = balanced_accuracy_score(y_male, y_pred)
      simple_ensemble_outer_loop_metrics_female_to_male['roc_aucs'][fold_idx] = roc_auc_score(y_male, y_prob)
      simple_ensemble_outer_loop_metrics_female_to_male['pr_aucs'][fold_idx] = average_precision_score(y_male, y_prob)
      simple_ensemble_metrics_female_to_male['all_true_labels'].extend(y_male)
      simple_ensemble_metrics_female_to_male['all_pred_probs'].extend(y_prob)

    # After all folds, average metrics for this repeat (like logreg_master.py)
    # Male to Female
    for model_type in MODELS:
      metrics_male_to_female[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics_male_to_female[model_type]['accuracies'])
      metrics_male_to_female[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics_male_to_female[model_type]['balanced_accuracies'])
      metrics_male_to_female[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics_male_to_female[model_type]['roc_aucs'])
      metrics_male_to_female[model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics_male_to_female[model_type]['pr_aucs'])
      metrics_male_to_female[model_type]['coefficients'][(repeat_idx*n_splits):(repeat_idx*n_splits)+n_splits, :] = outer_loop_metrics_male_to_female[model_type]['coefs']

    ensemble_metrics_male_to_female['accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_male_to_female['accuracies'])
    ensemble_metrics_male_to_female['balanced_accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_male_to_female['balanced_accuracies'])
    ensemble_metrics_male_to_female['roc_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_male_to_female['roc_aucs'])
    ensemble_metrics_male_to_female['pr_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_male_to_female['pr_aucs'])
    ensemble_metrics_male_to_female['coefficients'][(repeat_idx*n_splits):(repeat_idx*n_splits)+n_splits, :] = ensemble_outer_loop_metrics_male_to_female['coefs']

    simple_ensemble_metrics_male_to_female['accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_male_to_female['accuracies'])
    simple_ensemble_metrics_male_to_female['balanced_accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_male_to_female['balanced_accuracies'])
    simple_ensemble_metrics_male_to_female['roc_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_male_to_female['roc_aucs'])
    simple_ensemble_metrics_male_to_female['pr_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_male_to_female['pr_aucs'])

    # Female to Male
    for model_type in MODELS:
      metrics_female_to_male[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics_female_to_male[model_type]['accuracies'])
      metrics_female_to_male[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics_female_to_male[model_type]['balanced_accuracies'])
      metrics_female_to_male[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics_female_to_male[model_type]['roc_aucs'])
      metrics_female_to_male[model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics_female_to_male[model_type]['pr_aucs'])
      metrics_female_to_male[model_type]['coefficients'][(repeat_idx*n_splits):(repeat_idx*n_splits)+n_splits, :] = outer_loop_metrics_female_to_male[model_type]['coefs']

    ensemble_metrics_female_to_male['accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_female_to_male['accuracies'])
    ensemble_metrics_female_to_male['balanced_accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_female_to_male['balanced_accuracies'])
    ensemble_metrics_female_to_male['roc_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_female_to_male['roc_aucs'])
    ensemble_metrics_female_to_male['pr_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics_female_to_male['pr_aucs'])
    ensemble_metrics_female_to_male['coefficients'][(repeat_idx*n_splits):(repeat_idx*n_splits)+n_splits, :] = ensemble_outer_loop_metrics_female_to_male['coefs']

    simple_ensemble_metrics_female_to_male['accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_female_to_male['accuracies'])
    simple_ensemble_metrics_female_to_male['balanced_accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_female_to_male['balanced_accuracies'])
    simple_ensemble_metrics_female_to_male['roc_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_female_to_male['roc_aucs'])
    simple_ensemble_metrics_female_to_male['pr_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics_female_to_male['pr_aucs'])

    # Progress reporting
    if (repeat_idx + 1) % 10 == 0 or repeat_idx == 0:
        elapsed = time.time() - repeat_start_time
        avg_time_per_repeat = elapsed / (repeat_idx + 1)
        remaining_repeats = num_repeats - (repeat_idx + 1)
        estimated_remaining = avg_time_per_repeat * remaining_repeats

        print(f"Finished repeat {repeat_idx + 1}/{num_repeats} "
              f"(Elapsed: {elapsed/60:.1f}m, Est. remaining: {estimated_remaining/60:.1f}m)")

  return (metrics_male_to_female, metrics_female_to_male,
          ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
          simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male)


if __name__ == "__main__":
  N_SPLITS = 5
  N_REPEATS = 100
  RANDOM_STATE = 42
  SAVE_RESULTS = True
  MODELS = ['SC', 'FC', 'demos']
  CONTROL_ONLY = False  # Set to False to include moderate group as well

  # Set random seed for reproducibility
  np.random.seed(RANDOM_STATE)
  random_ints = np.random.randint(0, 1000, N_REPEATS)

  if CONTROL_ONLY:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  # Load male data
  X_dict_male = {
      'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}_male.npy'),
      'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}_male.npy'),
      'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}_male.npy')
  }
  y_male = np.load(f'data/training_data/aligned/y_aligned_{file_name}_male.npy')
  site_male = np.load(f'data/training_data/aligned/site_location_{file_name}_male.npy')

  # Load female data
  X_dict_female = {
      'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}_female.npy'),
      'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}_female.npy'),
      'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}_female.npy')
  }
  y_female = np.load(f'data/training_data/aligned/y_aligned_{file_name}_female.npy')
  site_female = np.load(f'data/training_data/aligned/site_location_{file_name}_female.npy')

  print("Male data shapes:", {model: X_dict_male[model].shape for model in MODELS}, "y:", y_male.shape)
  print("Female data shapes:", {model: X_dict_female[model].shape for model in MODELS}, "y:", y_female.shape)
  print(f"\nNumber of male subjects: {len(y_male)}")
  print(f"Number of female subjects: {len(y_female)}")
  print(f"Male class distribution: {np.bincount(y_male)} (class 0: {np.sum(y_male == 0)}, class 1: {np.sum(y_male == 1)})")
  print(f"Female class distribution: {np.bincount(y_female)} (class 0: {np.sum(y_female == 0)}, class 1: {np.sum(y_female == 1)})")
  print("\n")

  print("Running cross-sex testing (train on one sex, test on the other)...")
  if CONTROL_ONLY:
    print("Baseline subjects with cahalan=='control' only")
  else:
    print("Baseline subjects with cahalan=='control' or cahalan=='moderate'")
  print(f"Training on {len(y_male)} males → Testing on {len(y_female)} females (500 iterations)")
  print(f"Training on {len(y_female)} females → Testing on {len(y_male)} males (500 iterations)")
  print("\n")

  start = time.time()
  (metrics_male_to_female, metrics_female_to_male,
   ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
   simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male) = cross_sex_testing_repeated(
      X_dict_male, y_male, site_male,
      X_dict_female, y_female, site_female,
      N_SPLITS, N_REPEATS, random_ints
  )
  end = time.time()
  print(f"\n{'='*60}")
  print(f"Cross-sex testing completed!")
  print(f"Total time: {(end - start)/60:.1f} minutes ({end - start:.1f} seconds)")
  print(f"Generated {N_SPLITS * N_REPEATS} test results per direction")
  print(f"{'='*60}\n")

  if SAVE_RESULTS:
    print("Saving results...")
    save_results(MODELS, metrics_male_to_female, metrics_female_to_male,
                ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
                simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male,
                CONTROL_ONLY)
    print("Results saved successfully\n")

  print("Creating report...")
  create_metrics_report(metrics_male_to_female, metrics_female_to_male,
                       ensemble_metrics_male_to_female, ensemble_metrics_female_to_male,
                       simple_ensemble_metrics_male_to_female, simple_ensemble_metrics_female_to_male,
                       N_SPLITS, N_REPEATS, CONTROL_ONLY, save_to_file=SAVE_RESULTS)
  print(f"Report created successfully at results/reports/logreg_metrics/logreg_cross_sex_repeated_{file_name}.txt\n")
