"""
Permutation test for logistic regression models.
Calculates p-values for the metric of the models (SC, FC, FCgsr, demographics, ensemble, simple ensemble) for the following groups:
- Control only
- Control moderate
- Control moderate (male only)
- Control moderate (female only)
"""

import numpy as np

# metric should be either 'balanced_accuracies' or 'roc_aucs'
METRIC = 'roc_aucs'
# METRIC = 'balanced_accuracies'
SAVE_RESULTS = True

SC_metrics_control = np.load(f'results/SC/logreg/control/logreg_SC_control_{METRIC}.npy')
FC_metrics_control = np.load(f'results/FC/logreg/control/logreg_FC_control_{METRIC}.npy')
FCgsr_metrics_control = np.load(f'results/FCgsr/logreg/control/logreg_FCgsr_control_{METRIC}.npy')
demos_metrics_control = np.load(f'results/demos/logreg/control/logreg_demos_control_{METRIC}.npy')
ensemble_metrics_control = np.load(f'results/ensemble/logreg/control/logreg_ensemble_control_{METRIC}.npy')
simple_ensemble_metrics_control = np.load(f'results/simple_ensemble/logreg/control/logreg_simple_ensemble_control_{METRIC}.npy')

SC_metrics_control_moderate = np.load(f'results/SC/logreg/control_moderate/logreg_SC_control_moderate_{METRIC}.npy')
FC_metrics_control_moderate = np.load(f'results/FC/logreg/control_moderate/logreg_FC_control_moderate_{METRIC}.npy')
FCgsr_metrics_control_moderate = np.load(f'results/FCgsr/logreg/control_moderate/logreg_FCgsr_control_moderate_{METRIC}.npy')
demos_metrics_control_moderate = np.load(f'results/demos/logreg/control_moderate/logreg_demos_control_moderate_{METRIC}.npy')
ensemble_metrics_control_moderate = np.load(f'results/ensemble/logreg/control_moderate/logreg_ensemble_control_moderate_{METRIC}.npy')
simple_ensemble_metrics_control_moderate = np.load(f'results/simple_ensemble/logreg/control_moderate/logreg_simple_ensemble_control_moderate_{METRIC}.npy')

SC_metrics_male = np.load(f'results/SC/logreg/control_moderate/logreg_SC_control_moderate_male_{METRIC}.npy')
FC_metrics_male = np.load(f'results/FC/logreg/control_moderate/logreg_FC_control_moderate_male_{METRIC}.npy')
FCgsr_metrics_male = np.load(f'results/FCgsr/logreg/control_moderate/logreg_FCgsr_control_moderate_male_{METRIC}.npy')
demos_metrics_male = np.load(f'results/demos/logreg/control_moderate/logreg_demos_control_moderate_male_{METRIC}.npy')
ensemble_metrics_male = np.load(f'results/ensemble/logreg/control_moderate/logreg_ensemble_control_moderate_male_{METRIC}.npy')
simple_ensemble_metrics_male = np.load(f'results/simple_ensemble/logreg/control_moderate/logreg_simple_ensemble_control_moderate_male_{METRIC}.npy')

SC_metrics_female = np.load(f'results/SC/logreg/control_moderate/logreg_SC_control_moderate_female_{METRIC}.npy')
FC_metrics_female = np.load(f'results/FC/logreg/control_moderate/logreg_FC_control_moderate_female_{METRIC}.npy')
FCgsr_metrics_female = np.load(f'results/FCgsr/logreg/control_moderate/logreg_FCgsr_control_moderate_female_{METRIC}.npy')
demos_metrics_female = np.load(f'results/demos/logreg/control_moderate/logreg_demos_control_moderate_female_{METRIC}.npy')
ensemble_metrics_female = np.load(f'results/ensemble/logreg/control_moderate/logreg_ensemble_control_moderate_female_{METRIC}.npy')
simple_ensemble_metrics_female = np.load(f'results/simple_ensemble/logreg/control_moderate/logreg_simple_ensemble_control_moderate_female_{METRIC}.npy')

# Median for permuted data 
# Can be found in results/reports/permutation_test/permuted_logreg_metrics_report_XXX.txt
permuted_median_values = {
    "balanced_accuracies": {
        # control only
        "SC_metrics_control": 0.5030280433078091,
        "FC_metrics_control": 0.4992367635474668,
        "FCgsr_metrics_control": 0.5025475743858738,
        "demos_metrics_control": 0.501839777449594,
        "ensemble_metrics_control": 0.5035165373727104,
        "simple_ensemble_metrics_control": 0.502312545838648,
        # control moderate
        "SC_metrics_control_moderate": 0.5,
        "FC_metrics_control_moderate": 0.501471168771277,
        "FCgsr_metrics_control_moderate": 0.500195945945946,
        "demos_metrics_control_moderate": 0.5,
        "ensemble_metrics_control_moderate": 0.5048853950112061,
        "simple_ensemble_metrics_control_moderate": 0.5010029542529543,
        # Median for permuted data (control moderate MALE only)
        "SC_metrics_male": 0.5,
        "FC_metrics_male": 0.5,
        "FCgsr_metrics_male": 0.5,
        "demos_metrics_male": 0.5,
        "ensemble_metrics_male": 0.5079306298155715,
        "simple_ensemble_metrics_male": 0.5,
        # Median for permuted data (control moderate FEMALE only)
        "SC_metrics_female": 0.5085797311348782,
        "FC_metrics_female": 0.5036703696630167,
        "FCgsr_metrics_female": 0.49834317182395926,
        "demos_metrics_female": 0.5046875,
        "ensemble_metrics_female": 0.5043435515701882,
        "simple_ensemble_metrics_female": 0.5072976614888379
    },
    "roc_aucs": {
        # control only
        "SC_metrics_control": 0.5087469580225839,
        "FC_metrics_control": 0.500027152393373,
        "FCgsr_metrics_control": 0.5091213294579002,
        "demos_metrics_control": 0.534745951607574,
        "ensemble_metrics_control": 0.5044947920974687,
        "simple_ensemble_metrics_control": 0.5074776970325199,
        # control moderate
        "SC_metrics_control_moderate": 0.5106690831113325,
        "FC_metrics_control_moderate": 0.5100064804783622,
        "FCgsr_metrics_control_moderate": 0.5070411482911483,
        "demos_metrics_control_moderate": 0.5286799336312314,
        "ensemble_metrics_control_moderate": 0.5115027386378503,
        "simple_ensemble_metrics_control_moderate": 0.5109309723941302,
        # Median for permuted data (control moderate MALE only)
        "SC_metrics_male": 0.514805834262356,
        "FC_metrics_male": 0.5033091787439613,
        "FCgsr_metrics_male": 0.5049266072092159,
        "demos_metrics_male": 0.5434142911701236,
        "ensemble_metrics_male": 0.513792270531401,
        "simple_ensemble_metrics_male": 0.5167112597547381,
        # Median for permuted data (control moderate FEMALE only)
        "SC_metrics_female": 0.5164084898665349,
        "FC_metrics_female": 0.5040521799183131,
        "FCgsr_metrics_female": 0.500451318673637,
        "demos_metrics_female": 0.537203555497511,
        "ensemble_metrics_female": 0.5034435478736949,
        "simple_ensemble_metrics_female": 0.5092993703720348
    }
}

# Control only p-values
SC_control_pvalue = np.sum(SC_metrics_control < permuted_median_values[METRIC]["SC_metrics_control"]) / len(SC_metrics_control)
FC_control_pvalue = np.sum(FC_metrics_control < permuted_median_values[METRIC]["FC_metrics_control"]) / len(FC_metrics_control)
FCgsr_control_pvalue = np.sum(FCgsr_metrics_control < permuted_median_values[METRIC]["FCgsr_metrics_control"]) / len(FCgsr_metrics_control)
demos_control_pvalue = np.sum(demos_metrics_control < permuted_median_values[METRIC]["demos_metrics_control"]) / len(demos_metrics_control)
ensemble_control_pvalue = np.sum(ensemble_metrics_control < permuted_median_values[METRIC]["ensemble_metrics_control"]) / len(ensemble_metrics_control)
simple_ensemble_control_pvalue = np.sum(simple_ensemble_metrics_control < permuted_median_values[METRIC]["simple_ensemble_metrics_control"]) / len(simple_ensemble_metrics_control)

# Control moderate p-values
SC_control_moderate_pvalue = np.sum(SC_metrics_control_moderate < permuted_median_values[METRIC]["SC_metrics_control_moderate"]) / len(SC_metrics_control_moderate)
FC_control_moderate_pvalue = np.sum(FC_metrics_control_moderate < permuted_median_values[METRIC]["FC_metrics_control_moderate"]) / len(FC_metrics_control_moderate)
FCgsr_control_moderate_pvalue = np.sum(FCgsr_metrics_control_moderate < permuted_median_values[METRIC]["FCgsr_metrics_control_moderate"]) / len(FCgsr_metrics_control_moderate)
demos_control_moderate_pvalue = np.sum(demos_metrics_control_moderate < permuted_median_values[METRIC]["demos_metrics_control_moderate"]) / len(demos_metrics_control_moderate)
ensemble_control_moderate_pvalue = np.sum(ensemble_metrics_control_moderate < permuted_median_values[METRIC]["ensemble_metrics_control_moderate"]) / len(ensemble_metrics_control_moderate)
simple_ensemble_control_moderate_pvalue = np.sum(simple_ensemble_metrics_control_moderate < permuted_median_values[METRIC]["simple_ensemble_metrics_control_moderate"]) / len(simple_ensemble_metrics_control_moderate)

# Control moderate male p-values
SC_male_pvalue = np.sum(SC_metrics_male < permuted_median_values[METRIC]["SC_metrics_male"]) / len(SC_metrics_male)
FC_male_pvalue = np.sum(FC_metrics_male < permuted_median_values[METRIC]["FC_metrics_male"]) / len(FC_metrics_male)
FCgsr_male_pvalue = np.sum(FCgsr_metrics_male < permuted_median_values[METRIC]["FCgsr_metrics_male"]) / len(FCgsr_metrics_male)
demos_male_pvalue = np.sum(demos_metrics_male < permuted_median_values[METRIC]["demos_metrics_male"]) / len(demos_metrics_male)
ensemble_male_pvalue = np.sum(ensemble_metrics_male < permuted_median_values[METRIC]["ensemble_metrics_male"]) / len(ensemble_metrics_male)
simple_ensemble_male_pvalue = np.sum(simple_ensemble_metrics_male < permuted_median_values[METRIC]["simple_ensemble_metrics_male"]) / len(simple_ensemble_metrics_male)

# Control moderate female p-values
SC_female_pvalue = np.sum(SC_metrics_female < permuted_median_values[METRIC]["SC_metrics_female"]) / len(SC_metrics_female)
FC_female_pvalue = np.sum(FC_metrics_female < permuted_median_values[METRIC]["FC_metrics_female"]) / len(FC_metrics_female)
FCgsr_female_pvalue = np.sum(FCgsr_metrics_female < permuted_median_values[METRIC]["FCgsr_metrics_female"]) / len(FCgsr_metrics_female)
demos_female_pvalue = np.sum(demos_metrics_female < permuted_median_values[METRIC]["demos_metrics_female"]) / len(demos_metrics_female)
ensemble_female_pvalue = np.sum(ensemble_metrics_female < permuted_median_values[METRIC]["ensemble_metrics_female"]) / len(ensemble_metrics_female)
simple_ensemble_female_pvalue = np.sum(simple_ensemble_metrics_female < permuted_median_values[METRIC]["simple_ensemble_metrics_female"]) / len(simple_ensemble_metrics_female)

report = [
    "Permutation test (p-values) for logistic regression models",
    f"Metric: {METRIC}",
    "Alpha: 0.05\n",

    "Control Only P-Values",
    f"SC: {SC_control_pvalue}",
    f"FC: {FC_control_pvalue}",
    f"FCgsr: {FCgsr_control_pvalue}",
    f"Demos: {demos_control_pvalue}",
    f"Ensemble: {ensemble_control_pvalue}",
    f"Simple Ensemble: {simple_ensemble_control_pvalue}\n",

    "Control Moderate P-Values",
    f"SC: {SC_control_moderate_pvalue}",
    f"FC: {FC_control_moderate_pvalue}",
    f"FCgsr: {FCgsr_control_moderate_pvalue}",
    f"Demos: {demos_control_moderate_pvalue}",
    f"Ensemble: {ensemble_control_moderate_pvalue}",
    f"Simple Ensemble: {simple_ensemble_control_moderate_pvalue}\n",
    
    "Control Moderate (Male Only) P-Values",
    f"SC: {SC_male_pvalue}",
    f"FC: {FC_male_pvalue}",
    f"FCgsr: {FCgsr_male_pvalue}",
    f"Demos: {demos_male_pvalue}",
    f"Ensemble: {ensemble_male_pvalue}",
    f"Simple Ensemble: {simple_ensemble_male_pvalue}\n",

    "Control Moderate (Female Only) P-Values",
    f"SC: {SC_female_pvalue}",
    f"FC: {FC_female_pvalue}",
    f"FCgsr: {FCgsr_female_pvalue}",
    f"Demos: {demos_female_pvalue}",
    f"Ensemble: {ensemble_female_pvalue}",
    f"Simple Ensemble: {simple_ensemble_female_pvalue}",
]

if SAVE_RESULTS:
    with open(f'results/reports/permutation_test/permutation_test_logreg_{METRIC}_pvalues.txt', 'w') as report_file:
        report_file.write("\n".join(report))
else:
    print("\n".join(report))