import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_curve(matrix_type, control_only, male, female, save_fig=False):
  """
  Plot ROC curve.

  Args:
  - true_labels: Aggregated true labels from nested cross-validation.
  - pred_probs: Aggregated predicted probabilities from nested cross-validation.
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

  true_labels = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
  pred_probs = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')

  fpr, tpr, _ = roc_curve(true_labels, pred_probs)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve for {matrix_type} matrix ({file_name}{sex})')
  plt.legend(loc="lower right")
  if save_fig:
    plt.savefig(f'figures/roc_curve/{file_name}/roc_curve_{matrix_type}_{file_name}{sex}.png')
  else:
    plt.show()

def plot_roc_curve_combined(matrix_types, control_only, male, female, save_fig=False):
  """
  Plot multiple ROC curves on the same plot.

  Args:
  - matrix_types: List of matrix types to plot.
  - control_only: Boolean indicating whether to use control only data.
  - male: Boolean indicating whether to filter for male.
  - female: Boolean indicating whether to filter for female.
  - save_fig: Boolean indicating whether to save the figure.
  """
  if control_only:
      file_name = 'control'
  else:
      file_name = 'control_moderate'
  
  if male:
      sex = '_male'
      title = ' (Male)'
  elif female:
      sex = '_female'
      title = ' (Female)'
  else:
      sex = ''
      title = ''

  plt.figure()

  for matrix_type in matrix_types:
    true_labels = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
    pred_probs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')
    roc_auc_scores = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_roc_aucs.npy')

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = np.mean(roc_auc_scores, axis=0)

    if matrix_type == 'demos':
      matrix_type = 'Demographics'
    elif matrix_type == 'simple_ensemble':
      matrix_type = 'Ensemble'

    plt.plot(fpr, tpr, lw=2, label=f'{matrix_type} (AUC = {roc_auc:.2f})')

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curves for Model Type{title}')
  plt.legend(loc="lower right")
  plt.grid(axis='y', linestyle='--', alpha=0.3)
  
  if save_fig:
      plt.savefig(f'figures/roc_curve/{file_name}/roc_curve_{file_name}{sex}.png')
  else:
      plt.show()

def plot_pr_curve_combined(matrix_types, control_only, male, female, save_fig=False):
  """
  Plot multiple Precision-Recall curves on the same plot.

  Args:
  - matrix_types: List of matrix types to plot.
  - control_only: Boolean indicating whether to use control only data.
  - male: Boolean indicating whether to filter for male.
  - female: Boolean indicating whether to filter for female.
  - save_fig: Boolean indicating whether to save the figure.
  """
  from sklearn.metrics import precision_recall_curve
  
  if control_only:
      file_name = 'control'
  else:
      file_name = 'control_moderate'
  
  if male:
      sex = '_male'
      title = ' (Male)'
  elif female:
      sex = '_female'
      title = ' (Female)'
  else:
      sex = ''
      title = ''

  plt.figure()

  # For showing the baseline (no-skill classifier)
  baseline_shown = False
  
  for matrix_type in matrix_types:
    true_labels = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
    pred_probs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')
    aucpr_scores = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pr_aucs.npy')

    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    aucpr = np.mean(aucpr_scores, axis=0)

    if matrix_type == 'demos':
      matrix_type = 'Demographics'
    elif matrix_type == 'simple_ensemble':
      matrix_type = 'Ensemble'

    plt.plot(recall, precision, lw=2, label=f'{matrix_type} (AUC PR = {aucpr:.2f})')
    
    # Add baseline (proportion of positive samples) as a horizontal line - only need to do this once
    if not baseline_shown:
      baseline = np.sum(true_labels) / len(true_labels)
      plt.axhline(y=baseline, color='navy', linestyle='--', alpha=0.8, label=f'Baseline (No Skill): {baseline:.2f}')
      baseline_shown = True

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title(f'Precision-Recall Curves for Model Type{title}')
  plt.legend(loc="best")
  plt.grid(axis='both', linestyle='--', alpha=0.3)
  
  if save_fig:
      # Make sure this directory exists
      plt.savefig(f'figures/pr_curve/pr_curve_{file_name}{sex}.png')
  else:
      plt.show()

def plot_roc_pr_combined_grid(matrix_types, control_only, save_fig=False):
  """
  Plot a 2x3 grid of ROC and PR curves.
  Top row: ROC curves for Combined, Male-only, Female-only
  Bottom row: PR curves for Combined, Male-only, Female-only

  Args:
  - matrix_types: List of matrix types to plot.
  - control_only: Boolean indicating whether to use control only data.
  - save_fig: Boolean indicating whether to save the figure.
  """
  if control_only:
      file_name = 'control'
  else:
      file_name = 'control_moderate'

  # Create 2x3 subplot grid
  fig, axes = plt.subplots(2, 3, figsize=(18, 12))
  plt.subplots_adjust(wspace=0.15, hspace=0.8)

  # Define configurations for each column: (male, female, title)
  configs = [
      (False, False, 'Combined (All Subjects)'),
      (True, False, 'Male-only'),
      (False, True, 'Female-only')
  ]

  # Plot ROC curves (top row)
  for col, (male, female, title) in enumerate(configs):
      ax = axes[0, col]

      if male:
          sex = '_male'
      elif female:
          sex = '_female'
      else:
          sex = ''

      for matrix_type in matrix_types:
          true_labels = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
          pred_probs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')
          roc_auc_scores = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_roc_aucs.npy')

          fpr, tpr, _ = roc_curve(true_labels, pred_probs)
          roc_auc = np.mean(roc_auc_scores, axis=0)

          label_name = matrix_type
          if matrix_type == 'demos':
              label_name = 'Demographics'
          elif matrix_type == 'simple_ensemble':
              label_name = 'Ensemble'

          ax.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {roc_auc:.2f})')

      # Add diagonal reference line
      ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel('False Positive Rate', fontsize=22)

      # Only add y-label to leftmost plot
      if col == 0:
          ax.set_ylabel('True Positive Rate', fontsize=22)

      ax.set_title(title, fontsize=22)
      ax.legend(loc="lower right", fontsize=14)
      ax.grid(axis='y', linestyle='--', alpha=0.3)
      ax.tick_params(axis='both', labelsize=18)

      # Only show y-axis tick labels on leftmost plot
      if col != 0:
          ax.set_yticklabels([])

  # Plot PR curves (bottom row)
  for col, (male, female, title) in enumerate(configs):
      ax = axes[1, col]

      if male:
          sex = '_male'
      elif female:
          sex = '_female'
      else:
          sex = ''

      baseline_shown = False

      for matrix_type in matrix_types:
          true_labels = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_true_labels.npy')
          pred_probs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pred_probs.npy')
          aucpr_scores = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_pr_aucs.npy')

          precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
          aucpr = np.mean(aucpr_scores, axis=0)

          label_name = matrix_type
          if matrix_type == 'demos':
              label_name = 'Demographics'
          elif matrix_type == 'simple_ensemble':
              label_name = 'Ensemble'

          ax.plot(recall, precision, lw=2, label=f'{label_name} (AUC PR = {aucpr:.2f})')

          # Add baseline only once
          if not baseline_shown:
              baseline = np.sum(true_labels) / len(true_labels)
              ax.axhline(y=baseline, color='navy', linestyle='--', alpha=0.8, label=f'Baseline (No Skill): {baseline:.2f}')
              baseline_shown = True

      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel('Recall', fontsize=22)

      # Only add y-label to leftmost plot
      if col == 0:
          ax.set_ylabel('Precision', fontsize=22)

      # No title for bottom row
      ax.legend(loc="lower right", fontsize=14)
      ax.grid(axis='both', linestyle='--', alpha=0.3)
      ax.tick_params(axis='both', labelsize=18)

      # Only show y-axis tick labels on leftmost plot
      if col != 0:
          ax.set_yticklabels([])

  # Adjust layout to ensure proper spacing
  plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, wspace=0.15, hspace=0.4)

  # Add row labels (A) and (B) at the top of each row, positioned to avoid y-axis labels
  fig.text(0.03, 0.97, '(A)', fontsize=28, fontweight='bold', va='top', ha='left')
  fig.text(0.03, 0.48, '(B)', fontsize=28, fontweight='bold', va='top', ha='left')

  if save_fig:
      plt.savefig(f'figures/roc_pr_grid/roc_pr_grid_{file_name}.png', dpi=300, bbox_inches='tight')
  else:
      plt.show()

if __name__ == "__main__":
  CONTROL_ONLY = False
  MALE = False
  FEMALE = False
  SAVE_FIG = True

  # plot_roc_curve('SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('demos', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  # plot_roc_curve('ensemble', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_roc_curve('simple_ensemble', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  # plot_roc_curve_combined(['SC', 'FC', 'demos', 'ensemble'], control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)
  # plot_pr_curve_combined(['SC', 'FC', 'demos', 'ensemble'], control_only=CONTROL_ONLY, male=MALE, female=FEMALE, save_fig=SAVE_FIG)

  # Generate 2x3 grid with ROC and PR curves
  plot_roc_pr_combined_grid(['SC', 'FC', 'demos', 'simple_ensemble'], control_only=CONTROL_ONLY, save_fig=SAVE_FIG)