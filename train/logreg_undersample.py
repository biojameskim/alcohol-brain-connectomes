"""
Name: logreg_undersample.py
Purpose: Perform logistic regression on connectivity matrices (SC, FC), demographics, and ensemble models
         with optional sex-balanced undersampling and sex-stratified analysis.

Key Configuration Flags:
  UNDERSAMPLE: If True, applies undersampling to balance classes
  SEX_STRATIFIED: If True, trains/tests separate models for males and females
  MALE/FEMALE: If either is True, analyzes only that sex (cannot both be True)
  CONTROL_ONLY: If True, uses only 'control' subjects; if False, uses 'control' and 'moderate'
  PERMUTE: If True, randomly permutes labels for permutation testing

Usage Examples:
  1. Sex-Stratified Analysis (Separate Male/Female Models):
     UNDERSAMPLE=True, SEX_STRATIFIED=True, MALE=False, FEMALE=False
     - Loads both male and female data
     - Undersamples training to balance N between sexes and classes within each sex
     - Trains separate models for males and females
     - Tests on full test sets (not undersampled) for each sex
     - Saves results with '_male' and '_female' suffixes

  2. Combined Analysis (Single Model on Both Sexes):
     UNDERSAMPLE=True, SEX_STRATIFIED=False, MALE=False, FEMALE=False
     - Loads both male and female data
     - Undersamples training to balance N between sexes and classes within each sex
     - Trains a single combined model on both sexes
     - Tests on combined test set
     - Saves results without sex suffix

  3. Single Sex Analysis:
     UNDERSAMPLE=True, SEX_STRATIFIED=False, MALE=True (or FEMALE=True)
     - Loads only male (or female) data
     - Undersamples training to balance classes
     - Trains model on that sex only
     - Saves results with '_male' (or '_female') suffix
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def scale_data(X_train, X_test):
  """
  [scale_data] scales the input data [X] using the StandardScaler.
  Uses the parameters from train data to standardize both train and test. 
  """
  scaler = StandardScaler()
  X_scaled_train = scaler.fit_transform(X_train)
  X_scaled_test = scaler.transform(X_test)

  return X_scaled_train, X_scaled_test

def save_results(models, metrics, ensemble_metrics, simple_ensemble_metrics, control_only, male, female, permute, undersample):
  """
  [save_results] saves the resulting metrics from the logistic regression to NumPy files.
  """
  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'
  
  if male:
    sex = '_male'
  elif female:
    sex = '_female'
  else:
    sex = ''

  # Add undersample indicator to file names
  sampling = ""
  if undersample:
    sampling = f"_undersampled_equalN"

  if permute:
    for model_type in models:
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_balanced_accuracies.npy', metrics[model_type]['balanced_accuracies'])
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_roc_aucs.npy', metrics[model_type]['roc_aucs'])
      np.save(f'results/{model_type}/permuted_logreg_{model_type}_{file_name}{sex}{sampling}_pr_aucs.npy', metrics[model_type]['pr_aucs'])
    
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', ensemble_metrics['balanced_accuracies'])
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', ensemble_metrics['roc_aucs'])
    np.save(f'results/ensemble/permuted_logreg_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', ensemble_metrics['pr_aucs'])

    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', simple_ensemble_metrics['balanced_accuracies'])
    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', simple_ensemble_metrics['roc_aucs'])
    np.save(f'results/simple_ensemble/permuted_logreg_simple_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', simple_ensemble_metrics['pr_aucs'])

  else:
    for model_type in models:
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_accuracies.npy', metrics[model_type]['accuracies'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_balanced_accuracies.npy', metrics[model_type]['balanced_accuracies'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_roc_aucs.npy', metrics[model_type]['roc_aucs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_pr_aucs.npy', metrics[model_type]['pr_aucs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_true_labels.npy', metrics[model_type]['all_true_labels'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_pred_probs.npy', metrics[model_type]['all_pred_probs'])
        np.save(f'results/{model_type}/logreg_{model_type}_{file_name}{sex}{sampling}_coefficients.npy', metrics[model_type]['coefficients'])
    
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_accuracies.npy', ensemble_metrics['accuracies'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', ensemble_metrics['balanced_accuracies'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', ensemble_metrics['roc_aucs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', ensemble_metrics['pr_aucs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_true_labels.npy', ensemble_metrics['all_true_labels'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_pred_probs.npy', ensemble_metrics['all_pred_probs'])
    np.save(f'results/ensemble/logreg_ensemble_{file_name}{sex}{sampling}_coefficients.npy', ensemble_metrics['coefficients'])

    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_accuracies.npy', simple_ensemble_metrics['accuracies'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_balanced_accuracies.npy', simple_ensemble_metrics['balanced_accuracies'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_roc_aucs.npy', simple_ensemble_metrics['roc_aucs'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_pr_aucs.npy', simple_ensemble_metrics['pr_aucs'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_true_labels.npy', simple_ensemble_metrics['all_true_labels'])
    np.save(f'results/simple_ensemble/logreg_simple_ensemble_{file_name}{sex}{sampling}_pred_probs.npy', simple_ensemble_metrics['all_pred_probs'])

def create_metrics_report(metrics, ensemble_metrics, simple_ensemble_metrics, num_splits, num_repeats, control_only, male, female, permute, undersample, save_to_file=False):
  """
  [create_metrics_report] creates a report of the metrics for all models from the logistic regression
  """
  if control_only:
    control_str = "Baseline subjects with cahalan=='control' only"
  else:
    control_str = "Baseline subjects with cahalan=='control' or cahalan=='moderate'"

  if male:
    sex_str = "Male subjects only"
  elif female:
    sex_str = "Female subjects only"
  else:
    sex_str = "All subjects"

  # Add undersample indicator to file names
  undersample_str = ""
  if undersample:
    if not male and not female:
      undersample_str = f"Using sex-balanced undersampling (equal N between males/females, balanced classes within each sex)"
    else:
      undersample_str = f"Using undersampling to achieve class balance"

  report_lines = [
    "Results for logistic regression on matrices and ensemble:\n",
    f"Number of Splits: {num_splits}",
    f"Number of Repeats: {num_repeats}\n",
    
    f"{control_str}",
    f"{sex_str}",
    f"{undersample_str}\n",

    "Shape of matrices:\n",
    f"SC: {metrics['SC']['coefficients'].shape}",
    f"FC: {metrics['FC']['coefficients'].shape}",
    # f"FCgsr: {metrics['FCgsr']['coefficients'].shape}",
    f"Demographics: {metrics['demos']['coefficients'].shape}\n",

    "Median metrics:\n",
    f"SC Median accuracy: {np.median(metrics['SC']['accuracies'])}",
    f"FC Median accuracy: {np.median(metrics['FC']['accuracies'])}",
    # f"FCgsr Median accuracy: {np.median(metrics['FCgsr']['accuracies'])}",
    f"Demographics Median accuracy: {np.median(metrics['demos']['accuracies'])}",
    f"Ensemble Median accuracy: {np.median(ensemble_metrics['accuracies'])}",
    f"Simple Ensemble Median accuracy: {np.median(simple_ensemble_metrics['accuracies'])}\n",

    f"SC Median balanced accuracy: {np.median(metrics['SC']['balanced_accuracies'])}",
    f"FC Median balanced accuracy: {np.median(metrics['FC']['balanced_accuracies'])}",
    # f"FCgsr Median balanced accuracy: {np.median(metrics['FCgsr']['balanced_accuracies'])}",
    f"Demographics Median balanced accuracy: {np.median(metrics['demos']['balanced_accuracies'])}",
    f"Ensemble Median balanced accuracy: {np.median(ensemble_metrics['balanced_accuracies'])}",
    f"Simple Ensemble Median balanced accuracy: {np.median(simple_ensemble_metrics['balanced_accuracies'])}\n",

    f"SC Median ROC AUC: {np.median(metrics['SC']['roc_aucs'])}",
    f"FC Median ROC AUC: {np.median(metrics['FC']['roc_aucs'])}",
    # f"FCgsr Median ROC AUC: {np.median(metrics['FCgsr']['roc_aucs'])}",
    f"Demographics Median ROC AUC: {np.median(metrics['demos']['roc_aucs'])}",
    f"Ensemble Median ROC AUC: {np.median(ensemble_metrics['roc_aucs'])}",
    f"Simple Ensemble Median ROC AUC: {np.median(simple_ensemble_metrics['roc_aucs'])}\n",

    f"SC Median PR AUC: {np.median(metrics['SC']['pr_aucs'])}",
    f"FC Median PR AUC: {np.median(metrics['FC']['pr_aucs'])}",
    # f"FCgsr Median PR AUC: {np.median(metrics['FCgsr']['pr_aucs'])}",
    f"Demographics Median PR AUC: {np.median(metrics['demos']['pr_aucs'])}",
    f"Ensemble Median PR AUC: {np.median(ensemble_metrics['pr_aucs'])}",
    f"Simple Ensemble Median PR AUC: {np.median(simple_ensemble_metrics['pr_aucs'])}\n",

    "Mean metrics:\n",
    "SC:",
    f"Mean accuracy: {np.mean(metrics['SC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['SC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['SC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['SC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['SC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['SC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['SC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['SC']['pr_aucs'])}\n",

    "FC:",
    f"Mean accuracy: {np.mean(metrics['FC']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['FC']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['FC']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['FC']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['FC']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['FC']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['FC']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['FC']['pr_aucs'])}\n",

    # "FCgsr:",
    # f"Mean accuracy: {np.mean(metrics['FCgsr']['accuracies'])}",
    # f"Std accuracy: {np.std(metrics['FCgsr']['accuracies'])}",
    # f"Mean balanced accuracy: {np.mean(metrics['FCgsr']['balanced_accuracies'])}",
    # f"Std balanced accuracy: {np.std(metrics['FCgsr']['balanced_accuracies'])}",
    # f"Mean ROC AUC: {np.mean(metrics['FCgsr']['roc_aucs'])}",
    # f"Std ROC AUC: {np.std(metrics['FCgsr']['roc_aucs'])}",
    # f"Mean PR AUC: {np.mean(metrics['FCgsr']['pr_aucs'])}",
    # f"Std PR AUC: {np.std(metrics['FCgsr']['pr_aucs'])}\n",

    "Demographics:",
    f"Mean accuracy: {np.mean(metrics['demos']['accuracies'])}",
    f"Std accuracy: {np.std(metrics['demos']['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(metrics['demos']['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(metrics['demos']['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(metrics['demos']['roc_aucs'])}",
    f"Std ROC AUC: {np.std(metrics['demos']['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(metrics['demos']['pr_aucs'])}",
    f"Std PR AUC: {np.std(metrics['demos']['pr_aucs'])}\n",

    "Ensemble:",
    f"Mean accuracy: {np.mean(ensemble_metrics['accuracies'])}",
    f"Std accuracy: {np.std(ensemble_metrics['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(ensemble_metrics['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(ensemble_metrics['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(ensemble_metrics['roc_aucs'])}",
    f"Std ROC AUC: {np.std(ensemble_metrics['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(ensemble_metrics['pr_aucs'])}",
    f"Std PR AUC: {np.std(ensemble_metrics['pr_aucs'])}\n",

    "Simple Ensemble:",
    f"Mean accuracy: {np.mean(simple_ensemble_metrics['accuracies'])}",
    f"Std accuracy: {np.std(simple_ensemble_metrics['accuracies'])}",
    f"Mean balanced accuracy: {np.mean(simple_ensemble_metrics['balanced_accuracies'])}",
    f"Std balanced accuracy: {np.std(simple_ensemble_metrics['balanced_accuracies'])}",
    f"Mean ROC AUC: {np.mean(simple_ensemble_metrics['roc_aucs'])}",
    f"Std ROC AUC: {np.std(simple_ensemble_metrics['roc_aucs'])}",
    f"Mean PR AUC: {np.mean(simple_ensemble_metrics['pr_aucs'])}",
    f"Std PR AUC: {np.std(simple_ensemble_metrics['pr_aucs'])}"
  ]

  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'
  
  if male:
    sex = '_male'
  elif female:
    sex = '_female'
  else:
    sex = ''

  # Add undersampling indicator to file names
  sampling = ""
  if undersample:
    sampling = f"_undersampled_equalN"  

  if save_to_file and permute:
    with open(f'results/reports/permutation_test/permuted_logreg_metrics_report_{file_name}{sex}{sampling}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  elif save_to_file and not permute:
    with open(f'results/reports/logreg_metrics/logreg_metrics_report_{file_name}{sex}{sampling}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))
  else:
    print("\n".join(report_lines))

def create_model_and_metrics(X_dict, y, site_data, num_splits, num_repeats, random_ints, permute, undersample=False, sex_data=None, sex_stratified=False):
  """
  [create_model_and_metrics] performs logistic regression on the matrices and ensemble model.
  Returns the metrics for each model and the ensemble model.

  Args:
    sex_data: Array indicating sex (0=male, 1=female). If provided with undersample=True,
              will enforce equal total N between males and females.
    sex_stratified: If True, train/test separate models for males and females after undersampling.
                    Returns tuple of (male_metrics, female_metrics) instead of single metrics dict.
  """
  # Initialize metrics dictionaries
  # When sex_stratified=True, we'll return separate metrics for males and females
  if sex_stratified and sex_data is not None:
    # Initialize separate metrics for males and females
    metrics_male = {}
    metrics_female = {}
    for model in MODELS:
      metrics_male[model] = {"accuracies": np.empty(num_repeats),
                        "balanced_accuracies": np.empty(num_repeats),
                        "roc_aucs": np.empty(num_repeats),
                        "pr_aucs": np.empty(num_repeats),
                        "coefficients": np.empty((num_repeats*num_splits, X_dict[model].shape[1])),
                        "all_true_labels": [],
                        "all_pred_probs": []}
      metrics_female[model] = {"accuracies": np.empty(num_repeats),
                          "balanced_accuracies": np.empty(num_repeats),
                          "roc_aucs": np.empty(num_repeats),
                          "pr_aucs": np.empty(num_repeats),
                          "coefficients": np.empty((num_repeats*num_splits, X_dict[model].shape[1])),
                          "all_true_labels": [],
                          "all_pred_probs": []}

    ensemble_metrics_male = {"accuracies": np.empty(num_repeats),
                        "balanced_accuracies": np.empty(num_repeats),
                        "roc_aucs": np.empty(num_repeats),
                        "pr_aucs": np.empty(num_repeats),
                        "coefficients": np.empty((num_repeats*num_splits, len(MODELS))),
                        "all_true_labels": [],
                        "all_pred_probs": []}
    ensemble_metrics_female = {"accuracies": np.empty(num_repeats),
                          "balanced_accuracies": np.empty(num_repeats),
                          "roc_aucs": np.empty(num_repeats),
                          "pr_aucs": np.empty(num_repeats),
                          "coefficients": np.empty((num_repeats*num_splits, len(MODELS))),
                          "all_true_labels": [],
                          "all_pred_probs": []}

    simple_ensemble_metrics_male = {"accuracies": np.empty(num_repeats),
                                "balanced_accuracies": np.empty(num_repeats),
                                "roc_aucs": np.empty(num_repeats),
                                "pr_aucs": np.empty(num_repeats),
                                "coefficients": None,
                                "all_true_labels": [],
                                "all_pred_probs": []}
    simple_ensemble_metrics_female = {"accuracies": np.empty(num_repeats),
                                  "balanced_accuracies": np.empty(num_repeats),
                                  "roc_aucs": np.empty(num_repeats),
                                  "pr_aucs": np.empty(num_repeats),
                                  "coefficients": None,
                                  "all_true_labels": [],
                                  "all_pred_probs": []}
  else:
    # Standard metrics (non-sex-stratified)
    metrics = {}
    for model in MODELS:
      metrics[model] = {"accuracies": np.empty(num_repeats),
                        "balanced_accuracies": np.empty(num_repeats),
                        "roc_aucs": np.empty(num_repeats),
                        "pr_aucs": np.empty(num_repeats),
                        "coefficients": np.empty((num_repeats*num_splits, X_dict[model].shape[1])),
                        "all_true_labels": [],
                        "all_pred_probs": []}

    ensemble_metrics = {"accuracies": np.empty(num_repeats),
                        "balanced_accuracies": np.empty(num_repeats),
                        "roc_aucs": np.empty(num_repeats),
                        "pr_aucs": np.empty(num_repeats),
                        "coefficients": np.empty((num_repeats*num_splits, len(MODELS))),
                        "all_true_labels": [],
                        "all_pred_probs": []}

    simple_ensemble_metrics = {"accuracies": np.empty(num_repeats),
                                "balanced_accuracies": np.empty(num_repeats),
                                "roc_aucs": np.empty(num_repeats),
                                "pr_aucs": np.empty(num_repeats),
                                "coefficients": None, # No coefficients for simple ensemble
                                "all_true_labels": [],
                                "all_pred_probs": []}

  C_values = np.logspace(-4, 4, 15)
  outer_loop_metrics = {}

  print(f"\nStarting {num_repeats} repeated cross-validation runs...")
  print("=" * 60)
  repeat_start_time = time.time()

  for repeat_idx in range(num_repeats):
    repeat_iter_start = time.time()
    outer_kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_ints[repeat_idx])

    for model_type in MODELS:
      outer_loop_metrics[model_type] = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits),
                                        "pr_aucs": np.empty(num_splits),
                                        "coefs": np.empty((num_splits, X_dict[model_type].shape[1]))}
    
    ensemble_outer_loop_metrics = {"accuracies": np.empty(num_splits),
                                    "balanced_accuracies": np.empty(num_splits),
                                    "roc_aucs": np.empty(num_splits),
                                    "pr_aucs": np.empty(num_splits),
                                    "coefs": np.empty((num_splits, len(MODELS)))}
    
    simple_ensemble_outer_loop_metrics = {"accuracies": np.empty(num_splits),
                                        "balanced_accuracies": np.empty(num_splits),
                                        "roc_aucs": np.empty(num_splits),
                                        "pr_aucs": np.empty(num_splits)}

    if permute:
      y = np.random.permutation(y)

    n_samples = len(y) # StratifiedKFold stratifies on y so X doesn't matter in split (hence np.zeros(n_samples) below)
    simple_site_data = np.argmax(site_data, axis=1) # reduce complexity by reducing site data to a single column
    stratification_key = [str(a) + '_' + str(b) for a, b in zip(y, simple_site_data)]

    for fold_idx, (train_index, test_index) in enumerate(outer_kf.split(np.zeros(n_samples), stratification_key)):
      base_models = [] # Store the base models for the ensemble model in this fold

      # Track original distribution of classes and sites for reporting
      if fold_idx == 0 and repeat_idx == 0:
        class0_count = np.sum(y[train_index] == 0)
        class1_count = np.sum(y[train_index] == 1)
        site_counts = np.bincount(np.argmax(site_data[train_index], axis=1))
        print(f"\n{'='*60}")
        print(f"FIRST FOLD (fold 0) - BEFORE UNDERSAMPLING")
        print(f"{'='*60}")
        print(f"Training data (from train_index split):")
        print(f"  Total N: {len(train_index)}")
        print(f"  Class 0: {class0_count} ({class0_count/len(train_index):.2%})")
        print(f"  Class 1: {class1_count} ({class1_count/len(train_index):.2%})")
        print(f"  Site distribution: {site_counts}")
        print(f"\nTest data (from test_index split):")
        print(f"  Total N: {len(test_index)}")
        print(f"{'='*60}")

      # If undersampling, do it ONCE for ALL models to ensure consistency
      if undersample:
          y_train_original = y[train_index]
          site_train = site_data[train_index]
          train_site_ids = np.argmax(site_train, axis=1)

          # Sex-balanced undersampling: ensure equal total N between males and females
          if sex_data is not None:
              sex_train = sex_data[train_index]

              # Count samples for each sex and class combination
              male_mask = sex_train == 0
              female_mask = sex_train == 1

              male_class0 = np.sum((sex_train == 0) & (y_train_original == 0))
              male_class1 = np.sum((sex_train == 0) & (y_train_original == 1))
              female_class0 = np.sum((sex_train == 1) & (y_train_original == 0))
              female_class1 = np.sum((sex_train == 1) & (y_train_original == 1))

              # Find the minimum minority class count across both sexes
              # This ensures we can balance classes within each sex
              min_count = min(male_class0, male_class1, female_class0, female_class1)

              if fold_idx == 0 and repeat_idx == 0:
                  print(f"\nSex-balanced undersampling - BEFORE:")
                  print(f"  Males (N={male_class0 + male_class1}):")
                  print(f"    Class 0: {male_class0}")
                  print(f"    Class 1: {male_class1}")
                  print(f"    Class balance: {male_class0/(male_class0 + male_class1):.2%} / {male_class1/(male_class0 + male_class1):.2%}")
                  print(f"  Females (N={female_class0 + female_class1}):")
                  print(f"    Class 0: {female_class0}")
                  print(f"    Class 1: {female_class1}")
                  print(f"    Class balance: {female_class0/(female_class0 + female_class1):.2%} / {female_class1/(female_class0 + female_class1):.2%}")
                  print(f"\n  Undersampling target:")
                  print(f"    Min count across all groups: {min_count}")
                  print(f"    Target N per sex: {min_count * 2}")
                  print(f"    Target class balance per sex: 50% / 50%")

              # Sample indices for each combination
              np.random.seed(random_ints[repeat_idx + fold_idx])

              male_class0_indices = np.where((sex_train == 0) & (y_train_original == 0))[0]
              male_class1_indices = np.where((sex_train == 0) & (y_train_original == 1))[0]
              female_class0_indices = np.where((sex_train == 1) & (y_train_original == 0))[0]
              female_class1_indices = np.where((sex_train == 1) & (y_train_original == 1))[0]

              # Randomly sample min_count from each group
              sampled_male_class0 = np.random.choice(male_class0_indices, min_count, replace=False)
              sampled_male_class1 = np.random.choice(male_class1_indices, min_count, replace=False)
              sampled_female_class0 = np.random.choice(female_class0_indices, min_count, replace=False)
              sampled_female_class1 = np.random.choice(female_class1_indices, min_count, replace=False)

              # Combine all sampled indices
              resampled_indices = np.concatenate([
                  sampled_male_class0, sampled_male_class1,
                  sampled_female_class0, sampled_female_class1
              ])

              y_train_resampled = y_train_original[resampled_indices]
              resampled_sites = train_site_ids[resampled_indices]
              resampled_sex = sex_train[resampled_indices]

              if fold_idx == 0 and repeat_idx == 0:
                  male_resampled = np.sum(resampled_sex == 0)
                  female_resampled = np.sum(resampled_sex == 1)
                  male_class0_after = np.sum((resampled_sex == 0) & (y_train_resampled == 0))
                  male_class1_after = np.sum((resampled_sex == 0) & (y_train_resampled == 1))
                  female_class0_after = np.sum((resampled_sex == 1) & (y_train_resampled == 0))
                  female_class1_after = np.sum((resampled_sex == 1) & (y_train_resampled == 1))

                  print(f"\n  After undersampling - TRAINING SET:")
                  print(f"  Males (N={male_resampled}):")
                  print(f"    Class 0: {male_class0_after}")
                  print(f"    Class 1: {male_class1_after}")
                  print(f"    Class balance: {male_class0_after/male_resampled:.2%} / {male_class1_after/male_resampled:.2%}")
                  print(f"  Females (N={female_resampled}):")
                  print(f"    Class 0: {female_class0_after}")
                  print(f"    Class 1: {female_class1_after}")
                  print(f"    Class balance: {female_class0_after/female_resampled:.2%} / {female_class1_after/female_resampled:.2%}")
                  print(f"  Total training N: {len(resampled_indices)}")
                  print(f"  Sex balance: Males={male_resampled/len(resampled_indices):.2%}, Females={female_resampled/len(resampled_indices):.2%}")
                  site_counts_after = np.bincount(resampled_sites)
                  print(f"  Resampled site distribution: {site_counts_after}")

          else:
              # Original undersampling logic (no sex balancing)
              X_temp = X_dict[MODELS[0]][train_index]

              # Create a feature matrix with sample indices as the first column
              # This trick allows us to track which samples are selected
              sample_indices = np.arange(len(y_train_original)).reshape(-1, 1)
              temp_X = np.hstack((sample_indices, X_temp, train_site_ids.reshape(-1, 1)))

              # Apply stratified undersampling
              rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_ints[repeat_idx + fold_idx])
              temp_X_resampled, y_train_resampled = rus.fit_resample(temp_X, y_train_original)

              # Extract the indices of the samples that were kept
              resampled_indices = temp_X_resampled[:, 0].astype(int)

              # For reporting
              resampled_sites = train_site_ids[resampled_indices]

              if fold_idx == 0 and repeat_idx == 0:
                  class0_count_after = np.sum(y_train_resampled == 0)
                  class1_count_after = np.sum(y_train_resampled == 1)
                  site_counts_after = np.bincount(resampled_sites)
                  print(f"Resampled training data: Class 0: {class0_count_after}, Class 1: {class1_count_after}")
                  print(f"Resampled site distribution: {site_counts_after}")

      # SEX-STRATIFIED TRAINING: Train/test males and females separately
      if sex_stratified and sex_data is not None and undersample:
          # After undersampling, we have resampled_indices and resampled_sex
          # Split by sex for training
          sex_train_resampled = sex_data[train_index][resampled_indices]
          sex_test = sex_data[test_index]

          # Get male and female indices in the resampled training set
          male_train_mask = sex_train_resampled == 0
          female_train_mask = sex_train_resampled == 1

          # Get male and female indices in the test set
          male_test_mask = sex_test == 0
          female_test_mask = sex_test == 1

          if fold_idx == 0 and repeat_idx == 0:
              # Calculate test set distributions
              male_test_class0 = np.sum((sex_test == 0) & (y[test_index] == 0))
              male_test_class1 = np.sum((sex_test == 0) & (y[test_index] == 1))
              female_test_class0 = np.sum((sex_test == 1) & (y[test_index] == 0))
              female_test_class1 = np.sum((sex_test == 1) & (y[test_index] == 1))

              print(f"\n{'='*60}")
              print(f"SEX-STRATIFIED TRAINING/TESTING:")
              print(f"{'='*60}")
              print(f"\n  TRAINING (undersampled):")
              print(f"    Males: N={np.sum(male_train_mask)}")
              print(f"    Females: N={np.sum(female_train_mask)}")
              print(f"\n  TEST SET (full, not undersampled):")
              print(f"    Males (N={np.sum(male_test_mask)}):")
              print(f"      Class 0: {male_test_class0}")
              print(f"      Class 1: {male_test_class1}")
              print(f"      Class balance: {male_test_class0/np.sum(male_test_mask):.2%} / {male_test_class1/np.sum(male_test_mask):.2%}")
              print(f"    Females (N={np.sum(female_test_mask)}):")
              print(f"      Class 0: {female_test_class0}")
              print(f"      Class 1: {female_test_class1}")
              print(f"      Class balance: {female_test_class0/np.sum(female_test_mask):.2%} / {female_test_class1/np.sum(female_test_mask):.2%}")
              print(f"    Total test N: {len(test_index)}")
              print(f"{'='*60}\n")

          # Initialize outer loop metrics for males and females
          if fold_idx == 0:
              for model_type in MODELS:
                  if 'male' not in outer_loop_metrics:
                      outer_loop_metrics['male'] = {}
                      outer_loop_metrics['female'] = {}
                  outer_loop_metrics['male'][model_type] = {"accuracies": np.empty(num_splits),
                                                             "balanced_accuracies": np.empty(num_splits),
                                                             "roc_aucs": np.empty(num_splits),
                                                             "pr_aucs": np.empty(num_splits),
                                                             "coefs": np.empty((num_splits, X_dict[model_type].shape[1]))}
                  outer_loop_metrics['female'][model_type] = {"accuracies": np.empty(num_splits),
                                                               "balanced_accuracies": np.empty(num_splits),
                                                               "roc_aucs": np.empty(num_splits),
                                                               "pr_aucs": np.empty(num_splits),
                                                               "coefs": np.empty((num_splits, X_dict[model_type].shape[1]))}

              outer_loop_metrics['male']['ensemble'] = {"accuracies": np.empty(num_splits),
                                                        "balanced_accuracies": np.empty(num_splits),
                                                        "roc_aucs": np.empty(num_splits),
                                                        "pr_aucs": np.empty(num_splits),
                                                        "coefs": np.empty((num_splits, len(MODELS)))}
              outer_loop_metrics['female']['ensemble'] = {"accuracies": np.empty(num_splits),
                                                          "balanced_accuracies": np.empty(num_splits),
                                                          "roc_aucs": np.empty(num_splits),
                                                          "pr_aucs": np.empty(num_splits),
                                                          "coefs": np.empty((num_splits, len(MODELS)))}

              outer_loop_metrics['male']['simple_ensemble'] = {"accuracies": np.empty(num_splits),
                                                               "balanced_accuracies": np.empty(num_splits),
                                                               "roc_aucs": np.empty(num_splits),
                                                               "pr_aucs": np.empty(num_splits)}
              outer_loop_metrics['female']['simple_ensemble'] = {"accuracies": np.empty(num_splits),
                                                                 "balanced_accuracies": np.empty(num_splits),
                                                                 "roc_aucs": np.empty(num_splits),
                                                                 "pr_aucs": np.empty(num_splits)}

          # Train models for MALES and FEMALES separately
          for sex_label, sex_name, train_mask, test_mask in [
              (0, 'male', male_train_mask, male_test_mask),
              (1, 'female', female_train_mask, female_test_mask)
          ]:
              base_models_sex = []

              for model_type in MODELS:
                  X_train_full = X_dict[model_type][train_index][resampled_indices]
                  X_test_full = X_dict[model_type][test_index]
                  y_train_full = y[train_index][resampled_indices]
                  y_test_full = y[test_index]

                  # Filter by sex
                  X_train_sex = X_train_full[train_mask]
                  y_train_sex = y_train_full[train_mask]
                  X_test_sex = X_test_full[test_mask]
                  y_test_sex = y_test_full[test_mask]

                  inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])

                  pipeline = Pipeline([
                      ('scaler', StandardScaler()),
                      ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
                  ])

                  pipeline.fit(X_train_sex, y_train_sex)
                  base_models_sex.append(pipeline)

                  y_pred = pipeline.predict(X_test_sex)
                  y_prob = pipeline.predict_proba(X_test_sex)[:, 1]

                  # Store in sex-specific metrics
                  if sex_name == 'male':
                      metrics_male[model_type]['all_true_labels'].extend(y_test_sex)
                      metrics_male[model_type]['all_pred_probs'].extend(y_prob)
                  else:
                      metrics_female[model_type]['all_true_labels'].extend(y_test_sex)
                      metrics_female[model_type]['all_pred_probs'].extend(y_prob)

                  accuracy = accuracy_score(y_test_sex, y_pred)
                  bal_acc = balanced_accuracy_score(y_test_sex, y_pred)
                  roc_auc = roc_auc_score(y_test_sex, y_prob)
                  pr_auc = average_precision_score(y_test_sex, y_prob)

                  outer_loop_metrics[sex_name][model_type]['accuracies'][fold_idx] = accuracy
                  outer_loop_metrics[sex_name][model_type]['balanced_accuracies'][fold_idx] = bal_acc
                  outer_loop_metrics[sex_name][model_type]['roc_aucs'][fold_idx] = roc_auc
                  outer_loop_metrics[sex_name][model_type]['pr_aucs'][fold_idx] = pr_auc
                  outer_loop_metrics[sex_name][model_type]['coefs'][fold_idx] = pipeline.named_steps['classifier'].coef_[0]

              # Ensemble for this sex
              ensemble_X_trains_sex = {}
              for model_type in MODELS:
                  ensemble_X_trains_sex[model_type] = X_dict[model_type][train_index][resampled_indices][train_mask]
              ensemble_y_train_sex = y[train_index][resampled_indices][train_mask]

              train_preds_sex = np.column_stack([
                  cross_val_predict(
                      Pipeline([
                          ('scaler', StandardScaler()),
                          ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
                      ]),
                      ensemble_X_trains_sex[model_type], ensemble_y_train_sex,
                      method='predict_proba', cv=inner_kf, n_jobs=-1
                  )[:, 1]
                  for model_type in MODELS
              ])

              ensemble_model_sex = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
              ensemble_model_sex.fit(train_preds_sex, ensemble_y_train_sex)

              test_preds_sex = np.column_stack([
                  base_models_sex[i].predict_proba(X_dict[model_type][test_index][test_mask])[:, 1]
                  for i, model_type in enumerate(MODELS)
              ])

              y_test_sex = y[test_index][test_mask]
              y_pred_ensemble = ensemble_model_sex.predict(test_preds_sex)
              y_prob_ensemble = ensemble_model_sex.predict_proba(test_preds_sex)[:, 1]

              if sex_name == 'male':
                  ensemble_metrics_male['all_true_labels'].extend(y_test_sex)
                  ensemble_metrics_male['all_pred_probs'].extend(y_prob_ensemble)
              else:
                  ensemble_metrics_female['all_true_labels'].extend(y_test_sex)
                  ensemble_metrics_female['all_pred_probs'].extend(y_prob_ensemble)

              accuracy = accuracy_score(y_test_sex, y_pred_ensemble)
              bal_acc = balanced_accuracy_score(y_test_sex, y_pred_ensemble)
              roc_auc = roc_auc_score(y_test_sex, y_prob_ensemble)
              pr_auc = average_precision_score(y_test_sex, y_prob_ensemble)

              outer_loop_metrics[sex_name]['ensemble']['accuracies'][fold_idx] = accuracy
              outer_loop_metrics[sex_name]['ensemble']['balanced_accuracies'][fold_idx] = bal_acc
              outer_loop_metrics[sex_name]['ensemble']['roc_aucs'][fold_idx] = roc_auc
              outer_loop_metrics[sex_name]['ensemble']['pr_aucs'][fold_idx] = pr_auc
              outer_loop_metrics[sex_name]['ensemble']['coefs'][fold_idx] = ensemble_model_sex.coef_[0]

              # Simple ensemble for this sex
              averaged_preds_sex = np.mean([
                  base_models_sex[i].predict_proba(X_dict[model_type][test_index][test_mask])[:, 1]
                  for i, model_type in enumerate(MODELS)
              ], axis=0)
              y_pred_simple = (averaged_preds_sex >= 0.5).astype(int)
              y_prob_simple = averaged_preds_sex

              if sex_name == 'male':
                  simple_ensemble_metrics_male['all_true_labels'].extend(y_test_sex)
                  simple_ensemble_metrics_male['all_pred_probs'].extend(y_prob_simple)
              else:
                  simple_ensemble_metrics_female['all_true_labels'].extend(y_test_sex)
                  simple_ensemble_metrics_female['all_pred_probs'].extend(y_prob_simple)

              accuracy = accuracy_score(y_test_sex, y_pred_simple)
              bal_acc = balanced_accuracy_score(y_test_sex, y_pred_simple)
              roc_auc = roc_auc_score(y_test_sex, y_prob_simple)
              pr_auc = average_precision_score(y_test_sex, y_prob_simple)

              outer_loop_metrics[sex_name]['simple_ensemble']['accuracies'][fold_idx] = accuracy
              outer_loop_metrics[sex_name]['simple_ensemble']['balanced_accuracies'][fold_idx] = bal_acc
              outer_loop_metrics[sex_name]['simple_ensemble']['roc_aucs'][fold_idx] = roc_auc
              outer_loop_metrics[sex_name]['simple_ensemble']['pr_aucs'][fold_idx] = pr_auc

      elif not sex_stratified:
          # STANDARD (NON-SEX-STRATIFIED) TRAINING
          for model_type in MODELS:
            X_train_original = X_dict[model_type][train_index]
            X_test = X_dict[model_type][test_index]
            y_train_original = y[train_index]
            y_test = y[test_index]

            # Use the resampled indices if undersampling
            if undersample:
                X_train = X_train_original[resampled_indices]
                y_train = y_train_original[resampled_indices]
            else:
                X_train = X_train_original
                y_train = y_train_original

            inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])

            # Create a pipeline that includes standardization
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
            ])

            pipeline.fit(X_train, y_train)
            base_models.append(pipeline)

            # Get predictions from the pipeline
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Collect true labels and predicted probabilities for ROC curve
            metrics[model_type]['all_true_labels'].extend(y_test)
            metrics[model_type]['all_pred_probs'].extend(y_prob)

            accuracy = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            outer_loop_metrics[model_type]['accuracies'][fold_idx] = accuracy
            outer_loop_metrics[model_type]['balanced_accuracies'][fold_idx] = bal_acc
            outer_loop_metrics[model_type]['roc_aucs'][fold_idx] = roc_auc
            outer_loop_metrics[model_type]['pr_aucs'][fold_idx] = pr_auc
            outer_loop_metrics[model_type]['coefs'][fold_idx] = pipeline.named_steps['classifier'].coef_[0]

          # Ensemble model START
          # For the ensemble model, use the same resampled indices
          if undersample:
              # We already have the resampled indices, use them to create training data for each model
              ensemble_X_trains = {model_type: X_dict[model_type][train_index][resampled_indices] for model_type in MODELS}
              ensemble_y_train = y[train_index][resampled_indices]
          else:
              # If not undersampling, use original data
              ensemble_X_trains = {model_type: X_dict[model_type][train_index] for model_type in MODELS}
              ensemble_y_train = y[train_index]

          # Generate out-of-fold predictions for training the meta-model
          train_preds = np.column_stack([
              cross_val_predict(
                  Pipeline([
                      ('scaler', StandardScaler()),
                      ('classifier', LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1))
                  ]),
                  ensemble_X_trains[model_type], ensemble_y_train,
                  method='predict_proba', cv=inner_kf, n_jobs=-1
              )[:, 1]
              for model_type in MODELS
          ])

          # Train meta-model on unbiased OOF predictions
          inner_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_ints[repeat_idx + fold_idx])
          ensemble_model = LogisticRegressionCV(penalty='l2', Cs=C_values, cv=inner_kf, max_iter=100, n_jobs=-1)
          ensemble_model.fit(train_preds, ensemble_y_train)

          # Use previously trained base models to generate test predictions
          y_test = y[test_index]
          test_preds = np.column_stack([
              base_models[i].predict_proba(X_dict[model_type][test_index])[:, 1]
              for i, model_type in enumerate(MODELS)
          ])

          # Generate predictions using meta-model
          y_pred = ensemble_model.predict(test_preds)
          y_prob = ensemble_model.predict_proba(test_preds)[:, 1]

          ensemble_metrics['all_true_labels'].extend(y_test)
          ensemble_metrics['all_pred_probs'].extend(y_prob)

          accuracy = accuracy_score(y_test, y_pred)
          bal_acc = balanced_accuracy_score(y_test, y_pred)
          roc_auc = roc_auc_score(y_test, y_prob)
          pr_auc = average_precision_score(y_test, y_prob)

          ensemble_outer_loop_metrics['accuracies'][fold_idx] = accuracy
          ensemble_outer_loop_metrics['balanced_accuracies'][fold_idx] = bal_acc
          ensemble_outer_loop_metrics['roc_aucs'][fold_idx] = roc_auc
          ensemble_outer_loop_metrics['pr_aucs'][fold_idx] = pr_auc
          ensemble_outer_loop_metrics['coefs'][fold_idx] = ensemble_model.coef_[0]
          # Ensemble model END

          # Simple ensemble model START
          # Average the test predictions
          averaged_preds = np.mean([
            base_models[i].predict_proba(X_dict[model_type][test_index])[:, 1]
            for i, model_type in enumerate(MODELS)
          ], axis=0)
          y_pred = (averaged_preds >= 0.5).astype(int)
          y_prob = averaged_preds

          simple_ensemble_metrics['all_true_labels'].extend(y_test)
          simple_ensemble_metrics['all_pred_probs'].extend(y_prob)

          accuracy = accuracy_score(y_test, y_pred)
          bal_acc = balanced_accuracy_score(y_test, y_pred)
          roc_auc = roc_auc_score(y_test, y_prob)
          pr_auc = average_precision_score(y_test, y_prob)

          simple_ensemble_outer_loop_metrics['accuracies'][fold_idx] = accuracy
          simple_ensemble_outer_loop_metrics['balanced_accuracies'][fold_idx] = bal_acc
          simple_ensemble_outer_loop_metrics['roc_aucs'][fold_idx] = roc_auc
          simple_ensemble_outer_loop_metrics['pr_aucs'][fold_idx] = pr_auc
          # Simple ensemble model END

    # Aggregate metrics after all folds for this repeat
    if sex_stratified and sex_data is not None:
        # Aggregate sex-stratified metrics
        for sex_name in ['male', 'female']:
            sex_metrics = metrics_male if sex_name == 'male' else metrics_female
            sex_ensemble = ensemble_metrics_male if sex_name == 'male' else ensemble_metrics_female
            sex_simple_ensemble = simple_ensemble_metrics_male if sex_name == 'male' else simple_ensemble_metrics_female

            for model_type in MODELS:
                sex_metrics[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name][model_type]['accuracies'])
                sex_metrics[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name][model_type]['balanced_accuracies'])
                sex_metrics[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name][model_type]['roc_aucs'])
                sex_metrics[model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name][model_type]['pr_aucs'])
                sex_metrics[model_type]['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = outer_loop_metrics[sex_name][model_type]['coefs']

            sex_ensemble['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['ensemble']['accuracies'])
            sex_ensemble['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['ensemble']['balanced_accuracies'])
            sex_ensemble['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['ensemble']['roc_aucs'])
            sex_ensemble['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['ensemble']['pr_aucs'])
            sex_ensemble['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = outer_loop_metrics[sex_name]['ensemble']['coefs']

            sex_simple_ensemble['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['simple_ensemble']['accuracies'])
            sex_simple_ensemble['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['simple_ensemble']['balanced_accuracies'])
            sex_simple_ensemble['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['simple_ensemble']['roc_aucs'])
            sex_simple_ensemble['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[sex_name]['simple_ensemble']['pr_aucs'])
    else:
        # Aggregate standard (non-sex-stratified) metrics
        for model_type in MODELS:
            metrics[model_type]['accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['accuracies'])
            metrics[model_type]['balanced_accuracies'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['balanced_accuracies'])
            metrics[model_type]['roc_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['roc_aucs'])
            metrics[model_type]['pr_aucs'][repeat_idx] = np.mean(outer_loop_metrics[model_type]['pr_aucs'])
            metrics[model_type]['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = outer_loop_metrics[model_type]['coefs']

        ensemble_metrics['accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['accuracies'])
        ensemble_metrics['balanced_accuracies'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['balanced_accuracies'])
        ensemble_metrics['roc_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['roc_aucs'])
        ensemble_metrics['pr_aucs'][repeat_idx] = np.mean(ensemble_outer_loop_metrics['pr_aucs'])
        ensemble_metrics['coefficients'][(repeat_idx*num_splits):(repeat_idx*num_splits)+num_splits, :] = ensemble_outer_loop_metrics['coefs']

        simple_ensemble_metrics['accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['accuracies'])
        simple_ensemble_metrics['balanced_accuracies'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['balanced_accuracies'])
        simple_ensemble_metrics['roc_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['roc_aucs'])
        simple_ensemble_metrics['pr_aucs'][repeat_idx] = np.mean(simple_ensemble_outer_loop_metrics['pr_aucs'])

    # Print progress every 10 repeats
    if (repeat_idx + 1) % 10 == 0 or (repeat_idx + 1) == num_repeats:
        repeat_iter_time = time.time() - repeat_iter_start
        elapsed_total = time.time() - repeat_start_time
        avg_time_per_repeat = elapsed_total / (repeat_idx + 1)
        estimated_remaining = avg_time_per_repeat * (num_repeats - repeat_idx - 1)

        progress_pct = (repeat_idx + 1) / num_repeats * 100

        print(f"[{repeat_idx + 1}/{num_repeats}] ({progress_pct:.1f}%) - "
              f"Avg time/repeat: {avg_time_per_repeat:.1f}s - "
              f"Elapsed: {elapsed_total/60:.1f}min - "
              f"ETA: {estimated_remaining/60:.1f}min")

  total_time = time.time() - repeat_start_time
  print("=" * 60)
  print(f"All {num_repeats} repeats completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
  print("=" * 60)

  # Return appropriate metrics based on mode
  if sex_stratified and sex_data is not None:
      return (metrics_male, ensemble_metrics_male, simple_ensemble_metrics_male,
              metrics_female, ensemble_metrics_female, simple_ensemble_metrics_female)
  else:
      return metrics, ensemble_metrics, simple_ensemble_metrics

if __name__ == "__main__":
  N_SPLITS = 5
  N_REPEATS = 100
  RANDOM_STATE = 42
  SAVE_RESULTS = True
#   MODELS = ['SC', 'FC', 'FCgsr', 'demos']
  MODELS = ['SC', 'FC', 'demos']
  PERMUTE = False # If True, permute the labels (for permutation test)
  print("Permute: ", PERMUTE)
  CONTROL_ONLY = False # Set to False to include moderate group as well

  # BOTH of these sex flags cannot be True at the same time
  MALE = False # If True, only include male subjects
  FEMALE = False # If True, only include female subjects

  UNDERSAMPLE = True
  print("Undersample: ", UNDERSAMPLE)

  # Sex-stratified analysis: train/test males and females separately after sex-balanced undersampling
  SEX_STRATIFIED = True # Only applies when UNDERSAMPLE=True and MALE=False and FEMALE=False
  print("Sex-stratified: ", SEX_STRATIFIED)

  # Set random seed for reproducibility
  np.random.seed(RANDOM_STATE)
  random_ints = np.random.randint(0, 1000, N_REPEATS + N_SPLITS)

  if CONTROL_ONLY:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  if MALE:
    sex = '_male'
  elif FEMALE:
    sex = '_female'
  else:
    sex = ''

  # Load data (No standardization yet)
  # Load both male and female data separately, then combine with sex labels
  if UNDERSAMPLE and not MALE and not FEMALE:
    # Load male data
    X_dict_male = {'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}_male.npy'),
                   'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}_male.npy'),
                   'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}_male.npy')}
    y_male = np.load(f'data/training_data/aligned/y_aligned_{file_name}_male.npy')
    site_data_male = np.load(f'data/training_data/aligned/site_location_{file_name}_male.npy')
    sex_male = np.zeros(len(y_male))  # 0 for male

    # Load female data
    X_dict_female = {'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}_female.npy'),
                     'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}_female.npy'),
                     'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}_female.npy')}
    y_female = np.load(f'data/training_data/aligned/y_aligned_{file_name}_female.npy')
    site_data_female = np.load(f'data/training_data/aligned/site_location_{file_name}_female.npy')
    sex_female = np.ones(len(y_female))  # 1 for female

    # Combine data
    X_dict = {model: np.vstack([X_dict_male[model], X_dict_female[model]]) for model in MODELS}
    y = np.concatenate([y_male, y_female])
    site_data = np.vstack([site_data_male, site_data_female])
    sex_data = np.concatenate([sex_male, sex_female])

    print("\n" + "="*60)
    print("DATA LOADED - FULL DATASET (before any splitting/undersampling)")
    print("="*60)
    print(f"\nMales: N={len(y_male)}")
    print(f"  Class 0 (control): {np.sum(y_male == 0)} ({np.sum(y_male == 0)/len(y_male):.2%})")
    print(f"  Class 1 (problem): {np.sum(y_male == 1)} ({np.sum(y_male == 1)/len(y_male):.2%})")
    print(f"\nFemales: N={len(y_female)}")
    print(f"  Class 0 (control): {np.sum(y_female == 0)} ({np.sum(y_female == 0)/len(y_female):.2%})")
    print(f"  Class 1 (problem): {np.sum(y_female == 1)} ({np.sum(y_female == 1)/len(y_female):.2%})")
    print(f"\nCombined Total: N={len(y)}")
    print(f"  Males: {len(y_male)} ({len(y_male)/len(y):.2%})")
    print(f"  Females: {len(y_female)} ({len(y_female)/len(y):.2%})")
    print("="*60 + "\n")
  else:
    # Original behavior for single sex or no undersampling
    X_dict = {'SC': np.load(f'data/training_data/aligned/X_SC_{file_name}{sex}.npy'),
              'FC': np.load(f'data/training_data/aligned/X_FC_{file_name}{sex}.npy'),
              'demos': np.load(f'data/training_data/aligned/X_demos_{file_name}{sex}.npy')}
    y = np.load(f'data/training_data/aligned/y_aligned_{file_name}{sex}.npy')
    site_data = np.load(f'data/training_data/aligned/site_location_{file_name}{sex}.npy')
    sex_data = None

  print({model: X_dict[model].shape for model in MODELS}, "y:", y.shape)
  print("\n")

  sampling = ""
  if UNDERSAMPLE:
    sampling = f"_undersampled"

  print("Running logistic regression on matrices and ensemble (stratified)...")
  if CONTROL_ONLY:
    print("Baseline subjects with cahalan=='control' only")
  else:
    print("Baseline subjects with cahalan=='control' or cahalan=='moderate'")

  if MALE:
    print("Male subjects only")
  elif FEMALE:
    print("Female subjects only")
  else:
    print("All subjects")

  if UNDERSAMPLE and sex_data is not None:
    print("Sex-balanced undersampling enabled: Equal total N between males and females")
  elif UNDERSAMPLE:
    print("Standard undersampling enabled: Class balance only")

  print("\n")

  # Sex-stratified analysis: Run separately for males and females with sex-balanced undersampling
  if SEX_STRATIFIED and UNDERSAMPLE and sex_data is not None:
    print("\n" + "=" * 60)
    print("Running SEX-STRATIFIED analysis with sex-balanced undersampling")
    print("Data will be undersampled to ensure equal N between sexes,")
    print("then models will be trained/tested separately for each sex.")
    print("=" * 60)

    start = time.time()
    results = create_model_and_metrics(X_dict=X_dict, y=y, site_data=site_data,
                                      num_splits=N_SPLITS, num_repeats=N_REPEATS,
                                      random_ints=random_ints, permute=PERMUTE,
                                      undersample=UNDERSAMPLE, sex_data=sex_data,
                                      sex_stratified=True)
    end = time.time()
    print(f"Sex-stratified analysis finished in {end - start} seconds\n")

    # Unpack results
    metrics_male, ensemble_metrics_male, simple_ensemble_metrics_male, \
    metrics_female, ensemble_metrics_female, simple_ensemble_metrics_female = results

    # Save and report male results
    print("Saving male results...")
    save_results(models=MODELS, metrics=metrics_male, ensemble_metrics=ensemble_metrics_male,
                 simple_ensemble_metrics=simple_ensemble_metrics_male, control_only=CONTROL_ONLY,
                 male=True, female=False, permute=PERMUTE, undersample=UNDERSAMPLE)
    print("Male results saved successfully\n")

    print("Creating male report...")
    create_metrics_report(metrics=metrics_male, ensemble_metrics=ensemble_metrics_male,
                         simple_ensemble_metrics=simple_ensemble_metrics_male, num_splits=N_SPLITS,
                         num_repeats=N_REPEATS, control_only=CONTROL_ONLY, male=True, female=False,
                         permute=PERMUTE, undersample=UNDERSAMPLE, save_to_file=SAVE_RESULTS)
    print(f"Male report created successfully\n")

    # Save and report female results
    print("Saving female results...")
    save_results(models=MODELS, metrics=metrics_female, ensemble_metrics=ensemble_metrics_female,
                 simple_ensemble_metrics=simple_ensemble_metrics_female, control_only=CONTROL_ONLY,
                 male=False, female=True, permute=PERMUTE, undersample=UNDERSAMPLE)
    print("Female results saved successfully\n")

    print("Creating female report...")
    create_metrics_report(metrics=metrics_female, ensemble_metrics=ensemble_metrics_female,
                         simple_ensemble_metrics=simple_ensemble_metrics_female, num_splits=N_SPLITS,
                         num_repeats=N_REPEATS, control_only=CONTROL_ONLY, male=False, female=True,
                         permute=PERMUTE, undersample=UNDERSAMPLE, save_to_file=SAVE_RESULTS)
    print(f"Female report created successfully\n")

  else:
    # Standard (non-sex-stratified) analysis
    start = time.time()
    metrics, ensemble_metrics, simple_ensemble_metrics = create_model_and_metrics(X_dict=X_dict, y=y, site_data=site_data, num_splits=N_SPLITS, num_repeats=N_REPEATS, random_ints=random_ints, permute=PERMUTE, undersample=UNDERSAMPLE, sex_data=sex_data, sex_stratified=False)
    end = time.time()
    print(f"Finished in {end - start} seconds\n")

    print("Saving results...")
    save_results(models=MODELS, metrics=metrics, ensemble_metrics=ensemble_metrics, simple_ensemble_metrics=simple_ensemble_metrics, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, permute=PERMUTE, undersample=UNDERSAMPLE)
    print("Results saved successfully\n")

    print("Creating report...")
    create_metrics_report(metrics=metrics, ensemble_metrics=ensemble_metrics, simple_ensemble_metrics=simple_ensemble_metrics, num_splits=N_SPLITS, num_repeats=N_REPEATS, control_only=CONTROL_ONLY, male=MALE, female=FEMALE, permute=PERMUTE, undersample=UNDERSAMPLE, save_to_file=SAVE_RESULTS)
    print(f"Report created successfully at results/reports/logreg_metrics/stratified_logreg_metrics_report_{file_name}{sex}{sampling}.txt\n")