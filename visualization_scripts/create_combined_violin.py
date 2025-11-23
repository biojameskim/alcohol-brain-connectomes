import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_combined_figure(control_only=False, save_fig=False):
    """
    Creates a 2x3 grid figure combining:
    - Row A: Overall violin plots for each metric (Balanced Acc, ROC AUC, PR AUC)
    - Row B: Sex-stratified split violin plots for each metric

    Parameters:
    control_only (bool): Whether to use only control data or control+moderate data
    save_fig (bool): Whether to save the figure to a file
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    metrics = ['balanced_accuracies', 'roc_aucs', 'pr_aucs']
    metric_names = ['Balanced Accuracy', 'ROC AUC', 'PR AUC']

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(26, 12))

    # Color palettes
    sex_palette = {'Male': 'skyblue', 'Female': 'lightcoral'}

    # First pass: collect all data to determine global y-axis limits
    all_metric_data = []
    for metric in metrics:
        SC_metric = np.load(f'results/SC/logreg_SC_{file_name}_{metric}.npy')
        FC_metric = np.load(f'results/FC/logreg_FC_{file_name}_{metric}.npy')
        demos_metric = np.load(f'results/demos/logreg_demos_{file_name}_{metric}.npy')
        ensemble_metric = np.load(f'results/ensemble/logreg_ensemble_{file_name}_{metric}.npy')

        SC_male = np.load(f'results/SC/logreg_SC_{file_name}_male_{metric}.npy')
        FC_male = np.load(f'results/FC/logreg_FC_{file_name}_male_{metric}.npy')
        demos_male = np.load(f'results/demos/logreg_demos_{file_name}_male_{metric}.npy')
        ensemble_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_{metric}.npy')

        SC_female = np.load(f'results/SC/logreg_SC_{file_name}_female_{metric}.npy')
        FC_female = np.load(f'results/FC/logreg_FC_{file_name}_female_{metric}.npy')
        demos_female = np.load(f'results/demos/logreg_demos_{file_name}_female_{metric}.npy')
        ensemble_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_{metric}.npy')

        all_metric_data.extend([
            SC_metric, FC_metric, demos_metric, ensemble_metric,
            SC_male, FC_male, demos_male, ensemble_male,
            SC_female, FC_female, demos_female, ensemble_female
        ])

    # Calculate global y-axis limits
    global_min = min(arr.min() for arr in all_metric_data)
    global_max = max(arr.max() for arr in all_metric_data)
    global_range = global_max - global_min
    y_min = max(0, global_min - 0.05 * global_range)
    y_max = min(1, global_max + 0.05 * global_range)

    # Calculate tick spacing
    y_range = y_max - y_min
    tick_spacing = 0.05 if y_range > 0.15 else 0.02

    for col_idx, (metric, desc) in enumerate(zip(metrics, metric_names)):
        # ============= ROW A: Overall violin plots =============
        ax_overall = axes[0, col_idx]

        # Load overall data (not sex-stratified)
        SC_metric = np.load(f'results/SC/logreg_SC_{file_name}_{metric}.npy')
        FC_metric = np.load(f'results/FC/logreg_FC_{file_name}_{metric}.npy')
        demos_metric = np.load(f'results/demos/logreg_demos_{file_name}_{metric}.npy')
        ensemble_metric = np.load(f'results/ensemble/logreg_ensemble_{file_name}_{metric}.npy')

        # Create DataFrame for overall plot
        data_overall = pd.DataFrame({
            'Scores': np.concatenate([SC_metric, FC_metric, demos_metric, ensemble_metric]),
            'Category': ['SC'] * len(SC_metric) + ['FC'] * len(FC_metric) +
                       ['Demographics'] * len(demos_metric) + ['Ensemble'] * len(ensemble_metric)
        })

        # Create overall violin plot
        sns.violinplot(x='Category', y='Scores', data=data_overall, palette='muted',
                      legend=False, ax=ax_overall)
        ax_overall.set_ylim(y_min, y_max)
        ax_overall.set_yticks(np.arange(np.ceil(y_min/tick_spacing)*tick_spacing, y_max, tick_spacing))
        ax_overall.grid(axis='y', linestyle='--', alpha=0.3)

        # Add title showing metric name for top row only
        ax_overall.set_title(desc, fontsize=24)
        ax_overall.set_xlabel('')
        ax_overall.set_ylabel('')

        # Adjust tick label sizes and rotate x-axis labels to prevent overlap
        ax_overall.tick_params(axis='x', labelsize=22, rotation=15)
        ax_overall.tick_params(axis='y', labelsize=22)

        # Only show y-axis labels on leftmost plot
        if col_idx != 0:
            ax_overall.set_yticklabels([])

        # ============= ROW B: Sex-stratified split violin plots =============
        ax_sex = axes[1, col_idx]

        # Load male data
        SC_male = np.load(f'results/SC/logreg_SC_{file_name}_male_{metric}.npy')
        FC_male = np.load(f'results/FC/logreg_FC_{file_name}_male_{metric}.npy')
        demos_male = np.load(f'results/demos/logreg_demos_{file_name}_male_{metric}.npy')
        ensemble_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_{metric}.npy')

        # Load female data
        SC_female = np.load(f'results/SC/logreg_SC_{file_name}_female_{metric}.npy')
        FC_female = np.load(f'results/FC/logreg_FC_{file_name}_female_{metric}.npy')
        demos_female = np.load(f'results/demos/logreg_demos_{file_name}_female_{metric}.npy')
        ensemble_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_{metric}.npy')

        # Create DataFrame for sex-stratified plot
        data_sex = pd.DataFrame({
            'Score': np.concatenate([
                SC_male, SC_female,
                FC_male, FC_female,
                demos_male, demos_female,
                ensemble_male, ensemble_female
            ]),
            'Model': np.concatenate([
                ['SC'] * len(SC_male), ['SC'] * len(SC_female),
                ['FC'] * len(FC_male), ['FC'] * len(FC_female),
                ['Demographics'] * len(demos_male), ['Demographics'] * len(demos_female),
                ['Ensemble'] * len(ensemble_male), ['Ensemble'] * len(ensemble_female)
            ]),
            'Sex': np.concatenate([
                ['Male'] * len(SC_male), ['Female'] * len(SC_female),
                ['Male'] * len(FC_male), ['Female'] * len(FC_female),
                ['Male'] * len(demos_male), ['Female'] * len(demos_female),
                ['Male'] * len(ensemble_male), ['Female'] * len(ensemble_female)
            ])
        })

        # Create split violin plot
        sns.violinplot(
            x='Model',
            y='Score',
            hue='Sex',
            data=data_sex,
            split=True,
            inner='quartile',
            palette=sex_palette,
            cut=0,
            linewidth=1.5,
            ax=ax_sex
        )

        # Add chance line
        ax_sex.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

        ax_sex.set_ylim(y_min, y_max)
        ax_sex.set_yticks(np.arange(np.ceil(y_min/tick_spacing)*tick_spacing, y_max, tick_spacing))
        ax_sex.grid(axis='y', linestyle='--', alpha=0.3)

        # Remove titles
        ax_sex.set_title('')
        ax_sex.set_xlabel('')
        ax_sex.set_ylabel('')

        # Adjust tick label sizes and rotate x-axis labels to prevent overlap
        ax_sex.tick_params(axis='x', labelsize=22, rotation=15)
        ax_sex.tick_params(axis='y', labelsize=22)

        # Only show y-axis labels on leftmost plot
        if col_idx != 0:
            ax_sex.set_yticklabels([])

        # Only show legend on leftmost plot
        if col_idx == 0:
            ax_sex.legend(title='Sex', loc='upper left', fontsize=16, title_fontsize=16)
        else:
            ax_sex.get_legend().remove()

        # Add mean markers for sex-stratified plot
        means = data_sex.groupby(['Model', 'Sex'])['Score'].mean().unstack()
        for i, model in enumerate(['SC', 'FC', 'Demographics', 'Ensemble']):
            if model in means.index:
                if 'Male' in means.columns:
                    ax_sex.plot(i-0.1, means.loc[model, 'Male'], 'o', color='black', markersize=8)
                if 'Female' in means.columns:
                    ax_sex.plot(i+0.1, means.loc[model, 'Female'], 'o', color='black', markersize=8)

    # Add row labels (A) and (B) at the top of each row, positioned to avoid y-axis labels
    fig.text(0.03, 0.97, '(A)', fontsize=28, fontweight='bold', va='top', ha='left')
    fig.text(0.03, 0.50, '(B)', fontsize=28, fontweight='bold', va='top', ha='left')

    plt.tight_layout(rect=[0.06, 0, 1, 1])  # Leave space for row labels

    if save_fig:
        import os
        os.makedirs(f'figures/combined_violin/{file_name}', exist_ok=True)
        plt.savefig(f'figures/combined_violin/{file_name}/combined_figure_{file_name}.png',
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: figures/combined_violin/{file_name}/combined_figure_{file_name}.png")
    else:
        plt.show()


if __name__ == '__main__':
    CONTROL_ONLY = False
    SAVE_FIG = True

    print("Generating combined figure...")
    create_combined_figure(control_only=CONTROL_ONLY, save_fig=SAVE_FIG)
    print("Done!")
