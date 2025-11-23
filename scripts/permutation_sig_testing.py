"""
Permutation test for logistic regression models.
Calculates p-values for the metric of the models (SC, FC, FCgsr, demographics, ensemble, simple ensemble) for the following groups:
- Control moderate
- Control moderate (male only)
- Control moderate (female only)
"""

import numpy as np
from itertools import permutations, product
import pandas as pd
from statsmodels.stats.multitest import multipletests

def get_real_metrics():
    metrics = {
        'bal_acc': {
            'combined': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_balanced_accuracies.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_male_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_male_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_male_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_male_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_male_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_male_balanced_accuracies.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_female_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_female_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_female_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_female_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_female_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_female_balanced_accuracies.npy')
            }
        },
        'roc_auc': {
            'combined': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_roc_aucs.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_male_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_male_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_male_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_male_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_male_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_male_roc_aucs.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_female_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_female_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_female_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_female_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_female_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_female_roc_aucs.npy')
            }
        },
        'roc_auc': {
            'combined': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_roc_aucs.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_male_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_male_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_male_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_male_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_male_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_male_roc_aucs.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_female_roc_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_female_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_female_roc_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_female_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_female_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_female_roc_aucs.npy')
            }
        },
        'pr_auc': {
            'combined': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_pr_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_pr_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_pr_aucs.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_male_pr_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_male_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_male_pr_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_male_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_male_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_male_pr_aucs.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/logreg_SC_control_moderate_female_pr_aucs.npy'),
                'FC': np.load(f'results/FC/logreg_FC_control_moderate_female_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/logreg_FCgsr_control_moderate_female_pr_aucs.npy'),
                'demos': np.load(f'results/demos/logreg_demos_control_moderate_female_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/logreg_ensemble_control_moderate_female_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/logreg_simple_ensemble_control_moderate_female_pr_aucs.npy')
            }
        }
    }

    return metrics

def get_permuted_metrics():
    metrics = {
        'bal_acc': {
            'combined': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_balanced_accuracies.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_male_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_male_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_male_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_male_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_male_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_male_balanced_accuracies.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_female_balanced_accuracies.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_female_balanced_accuracies.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_female_balanced_accuracies.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_female_balanced_accuracies.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_female_balanced_accuracies.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_female_balanced_accuracies.npy')
            }
        },
        'roc_auc': {
            'combined': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_roc_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_roc_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_roc_aucs.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_male_roc_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_male_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_male_roc_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_male_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_male_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_male_roc_aucs.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_female_roc_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_female_roc_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_female_roc_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_female_roc_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_female_roc_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_female_roc_aucs.npy')
            }
        },
        'pr_auc': {
            'combined': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_pr_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_pr_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_pr_aucs.npy')
            },
            'male': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_male_pr_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_male_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_male_pr_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_male_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_male_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_male_pr_aucs.npy')
            },
            'female': {
                'SC': np.load(f'results/SC/permuted_logreg_SC_control_moderate_female_pr_aucs.npy'),
                'FC': np.load(f'results/FC/permuted_logreg_FC_control_moderate_female_pr_aucs.npy'),
                'FCgsr': np.load(f'results/FCgsr/permuted_logreg_FCgsr_control_moderate_female_pr_aucs.npy'),
                'demos': np.load(f'results/demos/permuted_logreg_demos_control_moderate_female_pr_aucs.npy'),
                'ensemble': np.load(f'results/ensemble/permuted_logreg_ensemble_control_moderate_female_pr_aucs.npy'),
                'simple_ensemble': np.load(f'results/simple_ensemble/permuted_logreg_simple_ensemble_control_moderate_female_pr_aucs.npy')
            }
        }
    }

    return metrics

def compute_pvalue(real, perm):
    obs = np.mean(real)
    K = perm.shape[0]
    return (np.sum(perm >= obs)) / (K)

def create_null_model_report(real_metrics, permuted_metrics, output_path='permutation_test_results.txt'):
    # Collect all p-values and their identifiers
    all_pvalues = []
    pvalue_identifiers = []  # To keep track of which p-value belongs to which test
    
    # Calculate all individual p-values first
    for metric in ['bal_acc', 'roc_auc', 'pr_auc']:
        for group in ['combined', 'male', 'female']:
            for model in real_metrics[metric][group]:
                real = real_metrics[metric][group][model]
                perm = permuted_metrics[metric][group][model]
                pval = compute_pvalue(real, perm)
                
                all_pvalues.append(pval)
                pvalue_identifiers.append((metric, group, model))
    
    # Apply FDR correction
    _, fdr_corrected_pvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
    
    # Create a dictionary mapping identifiers to both raw and corrected p-values
    results = {}
    for i, (metric, group, model) in enumerate(pvalue_identifiers):
        if (metric, group) not in results:
            results[(metric, group)] = {}
        
        results[(metric, group)][model] = {
            'raw_pvalue': all_pvalues[i],
            'fdr_pvalue': fdr_corrected_pvals[i]
        }
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write("Permutation Test Results with FDR Correction\n")
        f.write("=========================================\n\n")
        f.write("{:<10} | {:<10} | {:<15} | {:<12} | {:<12}\n".format(
            "Metric", "Group", "Model", "Raw p-value", "FDR p-value"))
        f.write("-" * 70 + "\n")
        
        # Process each combination
        for metric in ['bal_acc', 'roc_auc', 'pr_auc']:
            if metric == 'bal_acc':
                metric_name = "Balanced Acc"
            elif metric == "roc_auc":
                metric_name = "ROC AUC"
            else:
                metric_name = "PR AUC"
            
            for group in ['combined', 'male', 'female']:
                for model in real_metrics[metric][group]:
                    raw_pval = results[(metric, group)][model]['raw_pvalue']
                    fdr_pval = results[(metric, group)][model]['fdr_pvalue']
                    
                    # Format the line
                    line = "{:<10} | {:<10} | {:<15} | {:.8f}".format(
                        metric_name, group, model, raw_pval)
                    
                    # Add significance indicators for raw p-value
                    if raw_pval < 0.001:
                        line += " ***"
                    elif raw_pval < 0.01:
                        line += " **"
                    elif raw_pval < 0.05:
                        line += " *"
                    else:
                        line += "    "  # Padding for alignment
                    
                    # Add FDR p-value
                    line += " | {:.8f}".format(fdr_pval)
                    
                    # Add significance indicators for FDR p-value
                    if fdr_pval < 0.001:
                        line += " ***"
                    elif fdr_pval < 0.01:
                        line += " **"
                    elif fdr_pval < 0.05:
                        line += " *"
                        
                    f.write(line + "\n")
            
            # Add a blank line between metric types
            f.write("\n")
        
        # Add a legend for significance
        f.write("\nSignificance levels:\n")
        f.write("* p < 0.05\n")
        f.write("** p < 0.01\n")
        f.write("*** p < 0.001\n\n")
        
        f.write("Note on multiple testing corrections:\n")
        f.write("- Raw p-value: Original uncorrected p-value\n")
        f.write("- FDR p-value: Benjamini-Hochberg False Discovery Rate corrected p-value\n")
        f.write("  Controls the expected proportion of false positives among all rejected nulls\n")
    
    print(f"Report with FDR correction saved to {output_path}")

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################

def pairwise_permutation_test(model_a_metrics, model_b_metrics, n_permutations=10000, alternative='two-sided'):
    """Permutation test with corrected p-value calculation"""
    observed_diff = np.mean(model_a_metrics) - np.mean(model_b_metrics)
    
    all_metrics = np.concatenate([model_a_metrics, model_b_metrics])
    n = len(model_a_metrics)
    
    permuted_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        permuted = np.random.permutation(all_metrics)
        perm_a, perm_b = permuted[:n], permuted[n:]
        permuted_diffs[i] = np.mean(perm_a) - np.mean(perm_b)
    
    # Corrected p-value calculations with +1 adjustment
    if alternative == 'two-sided':
        p_value = (np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff))) / (n_permutations)
    elif alternative == 'greater':
        p_value = (np.sum(permuted_diffs >= observed_diff)) / (n_permutations)
    elif alternative == 'less':
        p_value = (np.sum(permuted_diffs <= observed_diff)) / (n_permutations)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return p_value

def run_all_pairwise_tests_with_corrections(real_metrics, group='combined', metric='bal_acc'):
    """
    Run pairwise tests with multiple comparison corrections
    
    Args:
        real_metrics: Dictionary of metrics
        group: 'combined', 'male', or 'female'
        metric: 'bal_acc' or 'roc_auc'
        model_pairs: Optional list of specific model pairs to test. If None, test all combinations.
    
    Returns:
        DataFrame with pairwise results and corrections
    """
    models = ['SC', 'FC', 'demos', 'ensemble']
    results = []
    
    # Store p-values for correction
    two_sided_pvals = []
    greater_pvals = []
    less_pvals = []
    model_pairs_list = []
    
    # Determine which pairs to evaluate
    pairs_to_evaluate = list(permutations(models, 2))
    
    # Get all pairwise combinations
    for model_a, model_b in pairs_to_evaluate:
        metrics_a = real_metrics[metric][group][model_a]
        metrics_b = real_metrics[metric][group][model_b]
        
        mean_a = np.mean(metrics_a)
        mean_b = np.mean(metrics_b)
        
        # Calculate uncorrected p-values
        p_greater = pairwise_permutation_test(metrics_a, metrics_b, alternative='greater')
        p_less = pairwise_permutation_test(metrics_a, metrics_b, alternative='less')
        p_two_sided = pairwise_permutation_test(metrics_a, metrics_b, alternative='two-sided')
        
        # Store for correction
        two_sided_pvals.append(p_two_sided)
        greater_pvals.append(p_greater)
        less_pvals.append(p_less)
        model_pairs_list.append((model_a, model_b))
        
        # Store result
        results.append({
            'Model A': model_a,
            'Model B': model_b,
            'Mean A': mean_a,
            'Mean B': mean_b,
            'Diff (A-B)': mean_a - mean_b,
            'p-value (two-sided)': p_two_sided,
            'p-value (A > B)': p_greater,
            'p-value (A < B)': p_less,
            'A > B': p_greater < 0.05,
            'A < B': p_less < 0.05,
            'A ≠ B': p_two_sided < 0.05
        })
    
    # Apply corrections if we have more than one comparison
    if len(two_sided_pvals) > 1:
        # Bonferroni correction
        _, bonf_two_sided = multipletests(two_sided_pvals, method='bonferroni')[:2]
        _, bonf_greater = multipletests(greater_pvals, method='bonferroni')[:2]
        _, bonf_less = multipletests(less_pvals, method='bonferroni')[:2]
        
        # FDR correction
        _, fdr_two_sided = multipletests(two_sided_pvals, method='fdr_bh')[:2]
        _, fdr_greater = multipletests(greater_pvals, method='fdr_bh')[:2]
        _, fdr_less = multipletests(less_pvals, method='fdr_bh')[:2]
        
        # Add corrected p-values to results
        for i, result in enumerate(results):
            # Bonferroni corrections
            result['p-value (two-sided, Bonferroni)'] = bonf_two_sided[i]
            result['p-value (A > B, Bonferroni)'] = bonf_greater[i]
            result['p-value (A < B, Bonferroni)'] = bonf_less[i]
            
            # FDR corrections
            result['p-value (two-sided, FDR)'] = fdr_two_sided[i]
            result['p-value (A > B, FDR)'] = fdr_greater[i]
            result['p-value (A < B, FDR)'] = fdr_less[i]
            
            # Significance with corrections
            result['A > B (Bonferroni)'] = bonf_greater[i] < 0.05
            result['A < B (Bonferroni)'] = bonf_less[i] < 0.05
            result['A ≠ B (Bonferroni)'] = bonf_two_sided[i] < 0.05
            
            result['A > B (FDR)'] = fdr_greater[i] < 0.05
            result['A < B (FDR)'] = fdr_less[i] < 0.05
            result['A ≠ B (FDR)'] = fdr_two_sided[i] < 0.05
    
    return pd.DataFrame(results)

def create_pairwise_report(real_metrics, output_path='pairwise_test_results_table.txt'):
    """Create a comprehensive tabular report with all model combinations and test directions"""
    with open(output_path, 'w') as f:
        f.write("Pairwise Model Comparison Results with Multiple Testing Corrections\n")
        f.write("===========================================================\n\n")
        
        # Create table header
        header = "{:<12} | {:<10} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15}\n".format(
            "Metric", "Group", "Model A", "Model B", "Raw p-value", "FDR p-value", "Bonferroni p-value")
        separator = "-" * 100 + "\n"
        
        # Process each test direction separately
        for test_direction in ["Two-sided", "A > B", "A < B"]:
            f.write(f"\n{test_direction} Tests\n")
            f.write("=" * len(f"{test_direction} Tests") + "\n\n")
            
            f.write(header)
            f.write(separator)
            
            # Process each combination
            for metric in ['bal_acc', 'roc_auc', 'pr_auc']:
                if metric == 'bal_acc':
                    metric_name = "Balanced Acc"
                elif metric == "roc_auc":
                    metric_name = "ROC AUC"
                else:
                    metric_name = "PR AUC"
                
                for group in ['combined', 'male', 'female']:
                    # Get all results at once (more efficient for corrections)
                    df = run_all_pairwise_tests_with_corrections(real_metrics, group, metric)
                    
                    # Process each row in the results
                    for _, row in df.iterrows():
                        model_a = row['Model A']
                        model_b = row['Model B']
                        
                        # Get p-values based on test direction
                        if test_direction == "Two-sided":
                            raw_pvalue = row['p-value (two-sided)']
                            fdr_pvalue = row.get('p-value (two-sided, FDR)', "N/A")
                            bonf_pvalue = row.get('p-value (two-sided, Bonferroni)', "N/A")
                        elif test_direction == "A > B":
                            raw_pvalue = row['p-value (A > B)']
                            fdr_pvalue = row.get('p-value (A > B, FDR)', "N/A")
                            bonf_pvalue = row.get('p-value (A > B, Bonferroni)', "N/A")
                        else:  # A < B
                            raw_pvalue = row['p-value (A < B)']
                            fdr_pvalue = row.get('p-value (A < B, FDR)', "N/A")
                            bonf_pvalue = row.get('p-value (A < B, Bonferroni)', "N/A")
                        
                        # Add significance indicators
                        raw_p_str = f"{raw_pvalue:.4f}"
                        if raw_pvalue < 0.05:
                            raw_p_str += "*"
                        
                        if isinstance(fdr_pvalue, float):
                            fdr_p_str = f"{fdr_pvalue:.4f}"
                            if fdr_pvalue < 0.05:
                                fdr_p_str += "*"
                        else:
                            fdr_p_str = fdr_pvalue
                        
                        if isinstance(bonf_pvalue, float):
                            bonf_p_str = f"{bonf_pvalue:.4f}"
                            if bonf_pvalue < 0.05:
                                bonf_p_str += "*"
                        else:
                            bonf_p_str = bonf_pvalue
                        
                        # Write table row
                        table_row = "{:<12} | {:<10} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15}\n".format(
                            metric_name, group, model_a, model_b, raw_p_str, fdr_p_str, bonf_p_str)
                        f.write(table_row)
                    
                    # Add a separator between groups
                    f.write(separator)
        
        # Add a legend for interpretation
        f.write("\n* p < 0.05 (statistically significant)\n\n")
        f.write("Test Interpretation:\n")
        f.write("- Two-sided: Tests whether models have different performance (any direction)\n")
        f.write("- A > B: Tests whether Model A is better than Model B\n")
        f.write("- A < B: Tests whether Model A is worse than Model B (or Model B is better than Model A)\n\n")
        f.write("Multiple Testing Corrections:\n")
        f.write("- Raw p-value: Uncorrected p-value\n")
        f.write("- FDR p-value: Benjamini-Hochberg corrected p-value (controls false discovery rate)\n")
        f.write("- Bonferroni p-value: Bonferroni corrected p-value (controls family-wise error rate)\n")
    
    print(f"Comprehensive pairwise report saved to {output_path}")


from itertools import product
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def run_all_cross_sex_pairwise_tests_with_corrections(real_metrics, metric='bal_acc', n_permutations=10000):
    """
    Compare each male model against each female model for a single metric.
    Returns a DataFrame with raw p-values, FDR, Bonferroni, and significance flags.
    """
    models = ['SC','FC','demos','ensemble']
    results = []

    # collect raw p-values for correction
    two_sided_p = []
    greater_p   = []
    less_p      = []

    # build all male×female pairs
    for m_model, f_model in product(models, models):
        a = real_metrics[metric]['male'][m_model]
        b = real_metrics[metric]['female'][f_model]
        mean_a, mean_b = np.mean(a), np.mean(b)

        p_g = pairwise_permutation_test(a, b, n_permutations, alternative='greater')
        p_l = pairwise_permutation_test(a, b, n_permutations, alternative='less')
        p_2 = pairwise_permutation_test(a, b, n_permutations, alternative='two-sided')

        two_sided_p.append(p_2)
        greater_p.append(p_g)
        less_p.append(p_l)

        results.append({
            'Metric': metric,
            'Model A': f"{m_model} (Male)",
            'Model B': f"{f_model} (Female)",
            'Mean A': mean_a,
            'Mean B': mean_b,
            'Diff (A−B)': mean_a - mean_b,
            'p-value (two-sided)': p_2,
            'p-value (A > B)': p_g,
            'p-value (A < B)': p_l
        })

    # apply corrections across all 16 two-sided, greater, less
    bf2 = multipletests(two_sided_p, method='bonferroni')[1]
    bfg = multipletests(greater_p,   method='bonferroni')[1]
    bfl = multipletests(less_p,      method='bonferroni')[1]
    fdr2= multipletests(two_sided_p, method='fdr_bh')[1]
    fdrg= multipletests(greater_p,   method='fdr_bh')[1]
    fdrl= multipletests(less_p,      method='fdr_bh')[1]

    for i, r in enumerate(results):
        r['p-value (two-sided, Bonferroni)'] = bf2[i]
        r['p-value (A > B, Bonferroni)']     = bfg[i]
        r['p-value (A < B, Bonferroni)']     = bfl[i]
        r['p-value (two-sided, FDR)']       = fdr2[i]
        r['p-value (A > B, FDR)']           = fdrg[i]
        r['p-value (A < B, FDR)']           = fdrl[i]
        r['A ≠ B (Bonferroni)'] = bf2[i] < 0.05
        r['A > B (Bonferroni)'] = bfg[i] < 0.05
        r['A < B (Bonferroni)'] = bfl[i] < 0.05
        r['A ≠ B (FDR)']       = fdr2[i] < 0.05
        r['A > B (FDR)']       = fdrg[i] < 0.05
        r['A < B (FDR)']       = fdrl[i] < 0.05

    return pd.DataFrame(results)


def create_cross_sex_pairwise_report(real_metrics, output_path='cross_sex_pairwise_test_results_table.txt'):
    """Create a comprehensive tabular report comparing every male‐model vs every female‐model."""
    with open(output_path, 'w') as f:
        f.write("Cross-Sex Model Comparison Results with Multiple Testing Corrections\n")
        f.write("===================================================================\n\n")
        
        # Create table header (no “Group” column)
        header = "{:<12} | {:<22} | {:<31} | {:<15} | {:<15} | {:<15}\n".format(
            "Metric", "Model A", "Model B", "Raw p-value", "FDR p-value", "Bonferroni p-value"
        )
        separator = "-" * 100 + "\n"
        
        # Process each test direction separately
        for test_direction in ["Two-sided", "A > B", "A < B"]:
            f.write(f"\n{test_direction} Tests\n")
            f.write("=" * len(f"{test_direction} Tests") + "\n\n")
            
            f.write(header)
            f.write(separator)
            
            # For each metric, get the 16 cross-sex comparisons
            for metric in ['bal_acc', 'roc_auc', 'pr_auc']:
                if metric == 'bal_acc':
                    metric_name = "Balanced Acc"
                elif metric == "roc_auc":
                    metric_name = "ROC AUC"
                else:
                    metric_name = "PR AUC"
                
                # run_all_cross_sex_pairwise_tests_with_corrections returns a DataFrame
                df = run_all_cross_sex_pairwise_tests_with_corrections(real_metrics, metric)
                
                for _, row in df.iterrows():
                    # choose the correct p-value columns
                    if test_direction == "Two-sided":
                        raw   = row['p-value (two-sided)']
                        fdr   = row['p-value (two-sided, FDR)']
                        bonf  = row['p-value (two-sided, Bonferroni)']
                    elif test_direction == "A > B":
                        raw   = row['p-value (A > B)']
                        fdr   = row['p-value (A > B, FDR)']
                        bonf  = row['p-value (A > B, Bonferroni)']
                    else:  # "A < B"
                        raw   = row['p-value (A < B)']
                        fdr   = row['p-value (A < B, FDR)']
                        bonf  = row['p-value (A < B, Bonferroni)']
                    
                    # format with significance star
                    def fmt(p): return f"{p:.4f}" + ("*" if p < 0.05 else "")
                    raw_s  = fmt(raw)
                    fdr_s  = fmt(fdr)
                    bonf_s = fmt(bonf)
                    
                    line = "{:<12} | {:<22} | {:<31} | {:<15} | {:<15} | {:<15}\n".format(
                        metric_name,
                        row['Model A'],
                        row['Model B'],
                        raw_s,
                        fdr_s,
                        bonf_s
                    )
                    f.write(line)
                
                f.write(separator)
        
        # legend
        f.write("\n* p < 0.05 (statistically significant)\n\n")
        f.write("Test Interpretation:\n")
        f.write("- Two-sided: any difference between Model A and Model B\n")
        f.write("- A > B: Model A > Model B\n")
        f.write("- A < B: Model A < Model B\n\n")
        f.write("Multiple Testing Corrections:\n")
        f.write("- Raw p-value: uncorrected\n")
        f.write("- FDR p-value: Benjamini–Hochberg\n")
        f.write("- Bonferroni p-value: Bonferroni\n")
    
    print(f"Cross-sex pairwise report saved to {output_path}")


def create_pairwise_report_non_tabular(real_metrics, output_path='pairwise_test_results.txt'):
    """Create comprehensive report with corrections"""
    with open(output_path, 'w') as f:
        f.write("Pairwise Model Comparison Results with Multiple Testing Corrections\n")
        f.write("===========================================================\n\n")
        
        for metric in ['bal_acc', 'roc_auc', 'pr_auc']:
            if metric == 'bal_acc':
                metric_name = "Balanced Acc"
            elif metric == "roc_auc":
                metric_name = "ROC AUC"
            else:
                metric_name = "PR AUC"
            f.write(f"\n{metric_name}\n")
            f.write("=" * len(metric_name) + "\n\n")
            
            for group in ['combined', 'male', 'female']:
                f.write(f"\n{group.capitalize()} Group\n")
                f.write("-" * (len(group) + 7) + "\n\n")
                
                # Get pairwise results with corrections
                df = run_all_pairwise_tests_with_corrections(real_metrics, group, metric)
                
                # Format results
                for _, row in df.iterrows():
                    f.write(f"{row['Model A']} vs {row['Model B']}:\n")
                    f.write(f"  Mean {row['Model A']}: {row['Mean A']:.4f}\n")
                    f.write(f"  Mean {row['Model B']}: {row['Mean B']:.4f}\n")
                    f.write(f"  Difference (A-B): {row['Diff (A-B)']:.4f}\n\n")
                    
                    # Uncorrected p-values
                    f.write("  Uncorrected p-values:\n")
                    f.write(f"    Two-sided (A ≠ B): {row['p-value (two-sided)']:.4f}")
                    if row['p-value (two-sided)'] < 0.001:
                        f.write(" ***\n")
                    elif row['p-value (two-sided)'] < 0.01:
                        f.write(" **\n")
                    elif row['p-value (two-sided)'] < 0.05:
                        f.write(" *\n")
                    else:
                        f.write(" (not significant)\n")
                    
                    f.write(f"    One-sided (A > B): {row['p-value (A > B)']:.4f}")
                    if row['p-value (A > B)'] < 0.05:
                        f.write(" *\n")
                    else:
                        f.write(" (not significant)\n")
                        
                    f.write(f"    One-sided (A < B): {row['p-value (A < B)']:.4f}")
                    if row['p-value (A < B)'] < 0.05:
                        f.write(" *\n")
                    else:
                        f.write(" (not significant)\n")
                    
                    # Include corrected p-values if available
                    if 'p-value (two-sided, Bonferroni)' in row:
                        # Bonferroni-corrected p-values
                        f.write("\n  Bonferroni-corrected p-values:\n")
                        f.write(f"    Two-sided (A ≠ B): {row['p-value (two-sided, Bonferroni)']:.4f}")
                        if row['p-value (two-sided, Bonferroni)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        f.write(f"    One-sided (A > B): {row['p-value (A > B, Bonferroni)']:.4f}")
                        if row['p-value (A > B, Bonferroni)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                            
                        f.write(f"    One-sided (A < B): {row['p-value (A < B, Bonferroni)']:.4f}")
                        if row['p-value (A < B, Bonferroni)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # FDR-corrected p-values
                        f.write("\n  FDR-corrected p-values:\n")
                        f.write(f"    Two-sided (A ≠ B): {row['p-value (two-sided, FDR)']:.4f}")
                        if row['p-value (two-sided, FDR)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        f.write(f"    One-sided (A > B): {row['p-value (A > B, FDR)']:.4f}")
                        if row['p-value (A > B, FDR)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                            
                        f.write(f"    One-sided (A < B): {row['p-value (A < B, FDR)']:.4f}")
                        if row['p-value (A < B, FDR)'] < 0.05:
                            f.write(" *\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # Conclusions
                        f.write("\n  Conclusions:\n")
                        
                        # Uncorrected
                        f.write("    Uncorrected: ")
                        if row['A > B'] and not row['A < B']:
                            f.write(f"{row['Model A']} is significantly better than {row['Model B']}\n")
                        elif row['A < B'] and not row['A > B']:
                            f.write(f"{row['Model A']} is significantly worse than {row['Model B']}\n")
                        elif row['A ≠ B']:
                            f.write(f"Models differ significantly but direction is ambiguous\n")
                        else:
                            f.write(f"No significant difference detected\n")
                        
                        # Bonferroni
                        f.write("    Bonferroni: ")
                        if row['A > B (Bonferroni)'] and not row['A < B (Bonferroni)']:
                            f.write(f"{row['Model A']} is significantly better than {row['Model B']}\n")
                        elif row['A < B (Bonferroni)'] and not row['A > B (Bonferroni)']:
                            f.write(f"{row['Model A']} is significantly worse than {row['Model B']}\n")
                        elif row['A ≠ B (Bonferroni)']:
                            f.write(f"Models differ significantly but direction is ambiguous\n")
                        else:
                            f.write(f"No significant difference detected\n")
                        
                        # FDR
                        f.write("    FDR: ")
                        if row['A > B (FDR)'] and not row['A < B (FDR)']:
                            f.write(f"{row['Model A']} is significantly better than {row['Model B']}\n")
                        elif row['A < B (FDR)'] and not row['A > B (FDR)']:
                            f.write(f"{row['Model A']} is significantly worse than {row['Model B']}\n")
                        elif row['A ≠ B (FDR)']:
                            f.write(f"Models differ significantly but direction is ambiguous\n")
                        else:
                            f.write(f"No significant difference detected\n")
                    else:
                        # Only uncorrected conclusion if no corrections
                        f.write("\n  Conclusion: ")
                        if row['A > B'] and not row['A < B']:
                            f.write(f"{row['Model A']} is significantly better than {row['Model B']}\n")
                        elif row['A < B'] and not row['A > B']:
                            f.write(f"{row['Model A']} is significantly worse than {row['Model B']}\n")
                        elif row['A ≠ B']:
                            f.write(f"Models differ significantly but direction is ambiguous\n")
                        else:
                            f.write(f"No significant difference detected\n")
                    
                    f.write("\n" + "-" * 40 + "\n\n")
            
        # Add a legend for significance
        f.write("\nSignificance levels:\n")
        f.write("* p < 0.05\n")
        f.write("** p < 0.01 (uncorrected only)\n")
        f.write("*** p < 0.001 (uncorrected only)\n")
        f.write("\nMultiple Testing Corrections:\n")
        f.write("- Bonferroni: Conservative correction that controls family-wise error rate\n")
        f.write("- FDR (Benjamini-Hochberg): Controls false discovery rate, more powerful than Bonferroni\n")
    
    print(f"Pairwise report with corrections saved to {output_path}")

if __name__ == "__main__":
    real_metrics = get_real_metrics()
    permuted_metrics = get_permuted_metrics()

    path_to_save_null = "results/reports/permutation_test/null_model_test_pvalues.txt"
    path_to_save_pairwise = "results/reports/permutation_test/pairwise_model_test_pvalues.txt"
    path_to_save_cross_sex_pairwise = "results/reports/permutation_test/pairwise_cross_sex_model_test_pvalues.txt"

    create_null_model_report(real_metrics, permuted_metrics, path_to_save_null)
    create_pairwise_report(real_metrics, path_to_save_pairwise)
    create_cross_sex_pairwise_report(real_metrics, path_to_save_cross_sex_pairwise)