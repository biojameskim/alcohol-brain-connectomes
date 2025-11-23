import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_full_comparison_plot(metric, control_only=False, save_fig=False):
    """
    Option 1: Shows within-sex and cross-sex performance side-by-side.
    Clearly demonstrates the generalization gap between same-sex and cross-sex predictions.

    Parameters:
    metric (str): One of 'accuracies', 'balanced_accuracies', 'roc_aucs', or 'pr_aucs'
    control_only (bool): Whether to use only control data or control+moderate data
    save_fig (bool): Whether to save the figure to a file
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    # Load within-sex data (train and test on same sex)
    SC_male_male = np.load(f'results/SC/logreg_SC_{file_name}_male_{metric}.npy')
    FC_male_male = np.load(f'results/FC/logreg_FC_{file_name}_male_{metric}.npy')
    demos_male_male = np.load(f'results/demos/logreg_demos_{file_name}_male_{metric}.npy')
    ensemble_male_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_{metric}.npy')

    SC_female_female = np.load(f'results/SC/logreg_SC_{file_name}_female_{metric}.npy')
    FC_female_female = np.load(f'results/FC/logreg_FC_{file_name}_female_{metric}.npy')
    demos_female_female = np.load(f'results/demos/logreg_demos_{file_name}_female_{metric}.npy')
    ensemble_female_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_{metric}.npy')

    # Load cross-sex data
    SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
    FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
    demos_male_female = np.load(f'results/demos/logreg_demos_{file_name}_male_to_female_{metric}.npy')
    ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')

    SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
    FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
    demos_female_male = np.load(f'results/demos/logreg_demos_{file_name}_female_to_male_{metric}.npy')
    ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

    # Create DataFrame with all conditions
    data = pd.DataFrame({
        'Score': np.concatenate([
            # Within-sex: Male
            SC_male_male, FC_male_male, demos_male_male, ensemble_male_male,
            # Within-sex: Female
            SC_female_female, FC_female_female, demos_female_female, ensemble_female_female,
            # Cross-sex: Male to Female
            SC_male_female, FC_male_female, demos_male_female, ensemble_male_female,
            # Cross-sex: Female to Male
            SC_female_male, FC_female_male, demos_female_male, ensemble_female_male,
        ]),
        'Model': np.concatenate([
            # Within-sex: Male
            ['SC'] * len(SC_male_male), ['FC'] * len(FC_male_male),
            ['Demographics'] * len(demos_male_male), ['Ensemble'] * len(ensemble_male_male),
            # Within-sex: Female
            ['SC'] * len(SC_female_female), ['FC'] * len(FC_female_female),
            ['Demographics'] * len(demos_female_female), ['Ensemble'] * len(ensemble_female_female),
            # Cross-sex: Male to Female
            ['SC'] * len(SC_male_female), ['FC'] * len(FC_male_female),
            ['Demographics'] * len(demos_male_female), ['Ensemble'] * len(ensemble_male_female),
            # Cross-sex: Female to Male
            ['SC'] * len(SC_female_male), ['FC'] * len(FC_female_male),
            ['Demographics'] * len(demos_female_male), ['Ensemble'] * len(ensemble_female_male),
        ]),
        'Condition': np.concatenate([
            # Within-sex: Male
            ['Male→Male'] * (len(SC_male_male) + len(FC_male_male) +
                            len(demos_male_male) + len(ensemble_male_male)),
            # Within-sex: Female
            ['Female→Female'] * (len(SC_female_female) + len(FC_female_female) +
                                len(demos_female_female) + len(ensemble_female_female)),
            # Cross-sex: Male to Female
            ['Male→Female'] * (len(SC_male_female) + len(FC_male_female) +
                              len(demos_male_female) + len(ensemble_male_female)),
            # Cross-sex: Female to Male
            ['Female→Male'] * (len(SC_female_male) + len(FC_female_male) +
                              len(demos_female_male) + len(ensemble_female_male)),
        ])
    })

    # Convert metric name to descriptive title
    if metric == "accuracies":
        desc = "Accuracy"
    elif metric == "balanced_accuracies":
        desc = "Balanced Accuracy"
    elif metric == "roc_aucs":
        desc = "ROC AUC"
    else:
        desc = "PR AUC"

    # Set up the plot with wider figure for better violin visibility
    fig, ax = plt.subplots(figsize=(24, 8))

    # Define color palette - darker for within-sex, lighter for cross-sex
    condition_palette = {
        'Male→Male': '#2E86AB',      # Dark blue
        'Female→Female': '#A23B72',   # Dark pink
        'Male→Female': '#A7C6DA',     # Light blue
        'Female→Male': '#E8B4D0'      # Light pink
    }

    # Create the violin plot with adjusted width
    sns.violinplot(
        x='Model',
        y='Score',
        hue='Condition',
        data=data,
        palette=condition_palette,
        inner='quartile',
        cut=0,
        linewidth=1.5,
        width=1.0,  # Make violins wider for better visibility
        ax=ax
    )

    # Calculate dynamic y-axis limits based on data
    data_min = data['Score'].min()
    data_max = data['Score'].max()
    data_range = data_max - data_min
    y_min = max(0, data_min - 0.05 * data_range)  # Add 5% padding below
    y_max = min(1, data_max + 0.05 * data_range)  # Add 5% padding above

    # Add a horizontal line at 0.5 (chance level)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Chance Level')

    # Customize the plot
    ax.set_title(f'{desc} Scores: Within-Sex vs Cross-Sex Generalization', fontsize=22)
    ax.set_xlabel('Model Type', fontsize=14)
    ax.set_ylabel(f'{desc} Score', fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.legend(title='Train→Test', loc='upper left', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Print y-axis range info
    print(f"Y-axis range: [{y_min:.4f}, {y_max:.4f}] (data range: [{data_min:.4f}, {data_max:.4f}])")

    # Print statistics
    print(f"\n{desc} Statistics:")
    print("=" * 80)
    for model in ['SC', 'FC', 'Demographics', 'Ensemble']:
        print(f"\n{model}:")
        for condition in ['Male→Male', 'Female→Female', 'Male→Female', 'Female→Male']:
            subset = data[(data['Model'] == model) & (data['Condition'] == condition)]
            if len(subset) > 0:
                print(f"  {condition:20s}: mean={subset['Score'].mean():.4f}, std={subset['Score'].std():.4f}")

    plt.tight_layout()

    # Save the figure if requested
    if save_fig:
        plt.savefig(f'figures/crosssex_comparison/{file_name}/full_comparison_{metric}_{file_name}.png',
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_crosssex_only_plot(metric, control_only=False, save_fig=False):
    """
    Option 2: Shows only cross-sex performance with clear labels.
    Focuses specifically on generalization across sexes.

    Parameters:
    metric (str): One of 'accuracies', 'balanced_accuracies', 'roc_aucs', or 'pr_aucs'
    control_only (bool): Whether to use only control data or control+moderate data
    save_fig (bool): Whether to save the figure to a file
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    # Load cross-sex data only
    SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
    FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
    demos_male_female = np.load(f'results/demos/logreg_demos_{file_name}_male_to_female_{metric}.npy')
    ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')

    SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
    FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
    demos_female_male = np.load(f'results/demos/logreg_demos_{file_name}_female_to_male_{metric}.npy')
    ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

    # Create DataFrame
    data = pd.DataFrame({
        'Score': np.concatenate([
            SC_male_female, FC_male_female, demos_male_female, ensemble_male_female,
            SC_female_male, FC_female_male, demos_female_male, ensemble_female_male,
        ]),
        'Model': np.concatenate([
            ['SC'] * len(SC_male_female), ['FC'] * len(FC_male_female),
            ['Demographics'] * len(demos_male_female), ['Ensemble'] * len(ensemble_male_female),
            ['SC'] * len(SC_female_male), ['FC'] * len(FC_female_male),
            ['Demographics'] * len(demos_female_male), ['Ensemble'] * len(ensemble_female_male),
        ]),
        'Direction': np.concatenate([
            ['Train Male\nTest Female'] * (len(SC_male_female) + len(FC_male_female) +
                                          len(demos_male_female) + len(ensemble_male_female)),
            ['Train Female\nTest Male'] * (len(SC_female_male) + len(FC_female_male) +
                                          len(demos_female_male) + len(ensemble_female_male)),
        ])
    })

    # Convert metric name to descriptive title
    if metric == "accuracies":
        desc = "Accuracy"
    elif metric == "balanced_accuracies":
        desc = "Balanced Accuracy"
    elif metric == "roc_aucs":
        desc = "ROC AUC"
    else:
        desc = "PR AUC"

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define color palette
    direction_palette = {
        'Train Male\nTest Female': '#6BAED6',  # Blue
        'Train Female\nTest Male': '#FD8D3C'   # Orange
    }

    # Create the violin plot
    sns.violinplot(
        x='Model',
        y='Score',
        hue='Direction',
        data=data,
        palette=direction_palette,
        split=False,  # Side-by-side, not split
        inner='quartile',
        cut=0,
        linewidth=1.5,
        ax=ax
    )

    # Calculate dynamic y-axis limits based on data
    data_min = data['Score'].min()
    data_max = data['Score'].max()
    data_range = data_max - data_min
    y_min = max(0, data_min - 0.05 * data_range)  # Add 5% padding below
    y_max = min(1, data_max + 0.05 * data_range)  # Add 5% padding above

    # Add a horizontal line at 0.5 (chance level)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Chance Level')

    # Get means for annotation
    means = data.groupby(['Model', 'Direction'])['Score'].mean().unstack()

    # Add mean markers
    models = ['SC', 'FC', 'Demographics', 'Ensemble']
    positions = np.arange(len(models))
    width = 0.4

    for i, model in enumerate(models):
        if model in means.index:
            # Train Male, Test Female
            if 'Train Male\nTest Female' in means.columns:
                ax.plot(i - width/2, means.loc[model, 'Train Male\nTest Female'],
                       'o', color='black', markersize=8, zorder=10)
            # Train Female, Test Male
            if 'Train Female\nTest Male' in means.columns:
                ax.plot(i + width/2, means.loc[model, 'Train Female\nTest Male'],
                       'o', color='black', markersize=8, zorder=10)

    # Customize the plot
    ax.set_title(f'{desc} Scores: Cross-Sex Generalization Performance', fontsize=22)
    ax.set_xlabel('Model Type', fontsize=14)
    ax.set_ylabel(f'{desc} Score', fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.legend(title='Generalization Direction', loc='upper left', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Print y-axis range info
    print(f"Y-axis range: [{y_min:.4f}, {y_max:.4f}] (data range: [{data_min:.4f}, {data_max:.4f}])")

    # Print statistics
    print(f"\n{desc} Cross-Sex Generalization Statistics:")
    print("=" * 80)
    for model in models:
        print(f"\n{model}:")
        for direction in ['Train Male\nTest Female', 'Train Female\nTest Male']:
            subset = data[(data['Model'] == model) & (data['Direction'] == direction)]
            if len(subset) > 0:
                mean_val = subset['Score'].mean()
                std_val = subset['Score'].std()
                print(f"  {direction.replace(chr(10), ' -> '):30s}: mean={mean_val:.4f}, std={std_val:.4f}")

    plt.tight_layout()

    # Save the figure if requested
    if save_fig:
        plt.savefig(f'figures/crosssex_comparison/{file_name}/crosssex_only_{metric}_{file_name}.png',
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_combined_full_comparison_plot(control_only=False, save_fig=False):
    """
    Creates a single figure with 3 vertically stacked subplots showing full comparison
    (within-sex vs cross-sex) for all 3 metrics.
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    metrics = ['balanced_accuracies', 'roc_aucs', 'pr_aucs']
    metric_names = ['Balanced Accuracy', 'ROC AUC', 'PR AUC']

    fig, axes = plt.subplots(3, 1, figsize=(24, 20))
    fig.suptitle('Within-Sex vs Cross-Sex Generalization', fontsize=22, y=0.995)

    # First pass: collect all data to determine global y-axis limits
    all_data = []
    for metric in metrics:
        SC_male_male = np.load(f'results/SC/logreg_SC_{file_name}_male_{metric}.npy')
        FC_male_male = np.load(f'results/FC/logreg_FC_{file_name}_male_{metric}.npy')
        ensemble_male_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_{metric}.npy')
        SC_female_female = np.load(f'results/SC/logreg_SC_{file_name}_female_{metric}.npy')
        FC_female_female = np.load(f'results/FC/logreg_FC_{file_name}_female_{metric}.npy')
        ensemble_female_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_{metric}.npy')
        SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
        FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
        ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')
        SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
        FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
        ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

        all_data.extend([
            SC_male_male, FC_male_male, ensemble_male_male,
            SC_female_female, FC_female_female, ensemble_female_female,
            SC_male_female, FC_male_female, ensemble_male_female,
            SC_female_male, FC_female_male, ensemble_female_male
        ])

    # Calculate global y-axis limits
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)
    global_range = global_max - global_min
    y_min = max(0, global_min - 0.05 * global_range)
    y_max = min(1, global_max + 0.05 * global_range)

    print(f"Global y-axis range for combined full comparison: [{y_min:.4f}, {y_max:.4f}]")

    for idx, (metric, desc) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Load all data for this metric (excluding Demographics for cross-sex comparison)
        SC_male_male = np.load(f'results/SC/logreg_SC_{file_name}_male_{metric}.npy')
        FC_male_male = np.load(f'results/FC/logreg_FC_{file_name}_male_{metric}.npy')
        ensemble_male_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_{metric}.npy')

        SC_female_female = np.load(f'results/SC/logreg_SC_{file_name}_female_{metric}.npy')
        FC_female_female = np.load(f'results/FC/logreg_FC_{file_name}_female_{metric}.npy')
        ensemble_female_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_{metric}.npy')

        SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
        FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
        ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')

        SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
        FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
        ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

        # Create DataFrame (Demographics removed - doesn't make sense for cross-sex comparison)
        data = pd.DataFrame({
            'Score': np.concatenate([
                SC_male_male, FC_male_male, ensemble_male_male,
                SC_female_female, FC_female_female, ensemble_female_female,
                SC_male_female, FC_male_female, ensemble_male_female,
                SC_female_male, FC_female_male, ensemble_female_male,
            ]),
            'Model': np.concatenate([
                ['SC'] * len(SC_male_male), ['FC'] * len(FC_male_male),
                ['Ensemble'] * len(ensemble_male_male),
                ['SC'] * len(SC_female_female), ['FC'] * len(FC_female_female),
                ['Ensemble'] * len(ensemble_female_female),
                ['SC'] * len(SC_male_female), ['FC'] * len(FC_male_female),
                ['Ensemble'] * len(ensemble_male_female),
                ['SC'] * len(SC_female_male), ['FC'] * len(FC_female_male),
                ['Ensemble'] * len(ensemble_female_male),
            ]),
            'Condition': np.concatenate([
                ['Male→Male'] * (len(SC_male_male) + len(FC_male_male) + len(ensemble_male_male)),
                ['Female→Female'] * (len(SC_female_female) + len(FC_female_female) + len(ensemble_female_female)),
                ['Male→Female'] * (len(SC_male_female) + len(FC_male_female) + len(ensemble_male_female)),
                ['Female→Male'] * (len(SC_female_male) + len(FC_female_male) + len(ensemble_female_male)),
            ])
        })

        condition_palette = {
            'Male→Male': '#2E86AB',
            'Female→Female': '#A23B72',
            'Male→Female': '#A7C6DA',
            'Female→Male': '#E8B4D0'
        }

        sns.violinplot(
            x='Model', y='Score', hue='Condition', data=data,
            palette=condition_palette, inner='quartile', cut=0,
            linewidth=1.5, width=0.8, ax=ax
        )

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)

        # Only show x-axis label on bottom plot
        if idx == 2:
            ax.set_xlabel('Model Type', fontsize=24)
            ax.tick_params(axis='x', labelsize=22)
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(f'{desc}', fontsize=24)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if idx == 0:
            ax.legend(title='Train→Test', loc='upper left', fontsize=15, title_fontsize=16)
        else:
            ax.get_legend().remove()

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'figures/crosssex_comparison/{file_name}/combined_full_comparison_{file_name}.png',
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: figures/crosssex_comparison/{file_name}/combined_full_comparison_{file_name}.png")
    else:
        plt.show()


def create_combined_crosssex_only_plot(control_only=False, save_fig=False):
    """
    Creates a single figure with 3 vertically stacked subplots showing cross-sex only
    performance for all 3 metrics.
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    metrics = ['balanced_accuracies', 'roc_aucs', 'pr_aucs']
    metric_names = ['Balanced Accuracy', 'ROC AUC', 'PR AUC']

    fig, axes = plt.subplots(3, 1, figsize=(18, 20))
    fig.suptitle('Cross-Sex Generalization Performance', fontsize=24, y=0.995)

    # First pass: collect all data to determine global y-axis limits
    all_data = []
    for metric in metrics:
        SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
        FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
        ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')
        SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
        FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
        ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

        all_data.extend([
            SC_male_female, FC_male_female, ensemble_male_female,
            SC_female_male, FC_female_male, ensemble_female_male
        ])

    # Calculate global y-axis limits
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)
    global_range = global_max - global_min
    y_min = max(0, global_min - 0.05 * global_range)
    y_max = min(1, global_max + 0.05 * global_range)

    print(f"Global y-axis range for combined cross-sex only: [{y_min:.4f}, {y_max:.4f}]")

    for idx, (metric, desc) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Load cross-sex data only (excluding Demographics)
        SC_male_female = np.load(f'results/SC/logreg_SC_{file_name}_male_to_female_{metric}.npy')
        FC_male_female = np.load(f'results/FC/logreg_FC_{file_name}_male_to_female_{metric}.npy')
        ensemble_male_female = np.load(f'results/ensemble/logreg_ensemble_{file_name}_male_to_female_{metric}.npy')

        SC_female_male = np.load(f'results/SC/logreg_SC_{file_name}_female_to_male_{metric}.npy')
        FC_female_male = np.load(f'results/FC/logreg_FC_{file_name}_female_to_male_{metric}.npy')
        ensemble_female_male = np.load(f'results/ensemble/logreg_ensemble_{file_name}_female_to_male_{metric}.npy')

        data = pd.DataFrame({
            'Score': np.concatenate([
                SC_male_female, FC_male_female, ensemble_male_female,
                SC_female_male, FC_female_male, ensemble_female_male,
            ]),
            'Model': np.concatenate([
                ['SC'] * len(SC_male_female), ['FC'] * len(FC_male_female),
                ['Ensemble'] * len(ensemble_male_female),
                ['SC'] * len(SC_female_male), ['FC'] * len(FC_female_male),
                ['Ensemble'] * len(ensemble_female_male),
            ]),
            'Direction': np.concatenate([
                ['Train Male\nTest Female'] * (len(SC_male_female) + len(FC_male_female) + len(ensemble_male_female)),
                ['Train Female\nTest Male'] * (len(SC_female_male) + len(FC_female_male) + len(ensemble_female_male)),
            ])
        })

        direction_palette = {
            'Train Male\nTest Female': '#6BAED6',
            'Train Female\nTest Male': '#FD8D3C'
        }

        sns.violinplot(
            x='Model', y='Score', hue='Direction', data=data,
            palette=direction_palette, split=False, inner='quartile',
            cut=0, linewidth=1.5, width=0.8, ax=ax
        )

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)

        # Add mean markers
        means = data.groupby(['Model', 'Direction'])['Score'].mean().unstack()
        models = ['SC', 'FC', 'Ensemble']
        width = 0.4
        for i, model in enumerate(models):
            if model in means.index:
                if 'Train Male\nTest Female' in means.columns:
                    ax.plot(i - width/2, means.loc[model, 'Train Male\nTest Female'],
                           'o', color='black', markersize=8, zorder=10)
                if 'Train Female\nTest Male' in means.columns:
                    ax.plot(i + width/2, means.loc[model, 'Train Female\nTest Male'],
                           'o', color='black', markersize=8, zorder=10)

        # Only show x-axis label on bottom plot
        if idx == 2:
            ax.set_xlabel('Model Type', fontsize=24)
            ax.tick_params(axis='x', labelsize=22)
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(f'{desc}', fontsize=24)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if idx == 0:
            ax.legend(title='Generalization Direction', loc='upper left', fontsize=15, title_fontsize=16)
        else:
            ax.get_legend().remove()

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'figures/crosssex_comparison/{file_name}/combined_crosssex_only_{file_name}.png',
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: figures/crosssex_comparison/{file_name}/combined_crosssex_only_{file_name}.png")
    else:
        plt.show()


if __name__ == '__main__':
    CONTROL_ONLY = False
    SAVE_FIG = True

    # Create output directory
    import os
    file_name = 'control' if CONTROL_ONLY else 'control_moderate'
    os.makedirs(f'figures/crosssex_comparison/{file_name}', exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Generating Combined Plots")
    print(f"{'='*80}")

    # Create combined plot 1: Full comparison with all 3 metrics
    print(f"\nGenerating Combined Full Comparison Plot (all 3 metrics)...")
    create_combined_full_comparison_plot(control_only=CONTROL_ONLY, save_fig=SAVE_FIG)
    plt.close()

    # Create combined plot 2: Cross-sex only with all 3 metrics
    print(f"\nGenerating Combined Cross-Sex Only Plot (all 3 metrics)...")
    create_combined_crosssex_only_plot(control_only=CONTROL_ONLY, save_fig=SAVE_FIG)
    plt.close()

    print(f"\n{'='*80}")
    print(f"All combined plots generated successfully!")
    print(f"{'='*80}")
