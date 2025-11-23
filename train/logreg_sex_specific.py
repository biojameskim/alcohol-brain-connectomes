"""
Name: logreg_sex_specific.py
Purpose: Evaluate sex-specific performance of logistic regression models trained on combined M+F data
"""

import numpy as np
import time
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(control_only=False, sex_specific=None):
    """Load data with optional sex filtering"""
    base_path = 'data/training_data/aligned/'
    file_suffix = '_control' if control_only else '_control_moderate'
    
    # Load main data
    X_dict = {
        'SC': np.load(f'{base_path}X_SC{file_suffix}.npy'),
        'FC': np.load(f'{base_path}X_FC{file_suffix}.npy'),
        'demos': np.load(f'{base_path}X_demos{file_suffix}.npy')
    }
    y = np.load(f'{base_path}y_aligned{file_suffix}.npy')
    
    # Extract sex information (Sex is the 4th column in demos data (0=female, 1=male))
    sex_data = X_dict['demos'][:, 3].astype(int)
    
    return X_dict, y, sex_data

def run_sex_specific_evaluation(X_dict, y, sex_data, site_data, random_states):
    """Main evaluation function with sex-specific splits"""
    # Initialize metrics storage (including ensemble model)
    all_models = MODELS + ['ensemble']
    
    metrics = {
        'male': {model: {
            'accuracies': np.zeros(N_REPEATS),
            'balanced_accuracies': np.zeros(N_REPEATS),
            'roc_aucs': np.zeros(N_REPEATS),
            'pr_aucs': np.zeros(N_REPEATS)
        } for model in all_models},
        'female': {model: {
            'accuracies': np.zeros(N_REPEATS),
            'balanced_accuracies': np.zeros(N_REPEATS),
            'roc_aucs': np.zeros(N_REPEATS),
            'pr_aucs': np.zeros(N_REPEATS)
        } for model in all_models}
    }

    for repeat in range(N_REPEATS):
        # Split male and female indices
        male_idx = np.where(sex_data == 1)[0]
        female_idx = np.where(sex_data == 0)[0]

        # Create stratification keys that combine outcome and site
        simple_site_data = np.argmax(site_data, axis=1)  # Convert one-hot to single column
        
        # Create stratification keys for males and females
        male_strat_keys = [f"{y[i]}_{simple_site_data[i]}" for i in male_idx]
        female_strat_keys = [f"{y[i]}_{simple_site_data[i]}" for i in female_idx]
        
        # Stratified splits that consider both outcome and site
        male_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, 
                                 random_state=random_states[repeat])
        female_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=random_states[repeat])

        # Store metrics for this repeat
        repeat_metrics = {
            'male': {model: [] for model in all_models},
            'female': {model: [] for model in all_models}
        }

        # Process male test folds
        for train_idx, test_idx in male_skf.split(male_idx, male_strat_keys):
            # Create train/test indices
            train_indices = np.concatenate([
                male_idx[train_idx],  # Other males
                female_idx            # All females
            ])
            test_indices = male_idx[test_idx]

            # Train/test split
            X_train = {m: X_dict[m][train_indices] for m in MODELS}
            X_test = {m: X_dict[m][test_indices] for m in MODELS}
            y_train, y_test = y[train_indices], y[test_indices]

            # Train and evaluate models (including ensemble)
            fold_metrics = evaluate_models_with_ensemble(X_train, X_test, y_train, y_test)
            
            # Store metrics
            for model in all_models:
                repeat_metrics['male'][model].append(fold_metrics[model])

        # Process female test folds
        for train_idx, test_idx in female_skf.split(female_idx, female_strat_keys):
            # Create train/test indices
            train_indices = np.concatenate([
                male_idx,              # All males
                female_idx[train_idx]  # Other females
            ])
            test_indices = female_idx[test_idx]

            # Train/test split
            X_train = {m: X_dict[m][train_indices] for m in MODELS}
            X_test = {m: X_dict[m][test_indices] for m in MODELS}
            y_train, y_test = y[train_indices], y[test_indices]

            # Train and evaluate models (including ensemble)
            fold_metrics = evaluate_models_with_ensemble(X_train, X_test, y_train, y_test)
            
            # Store metrics
            for model in all_models:
                repeat_metrics['female'][model].append(fold_metrics[model])

        # Average across folds for this repeat
        for sex in ['male', 'female']:
            for model in all_models:
                metrics[sex][model]['accuracies'][repeat] = np.mean(
                    [m['accuracy'] for m in repeat_metrics[sex][model]])
                metrics[sex][model]['balanced_accuracies'][repeat] = np.mean(
                    [m['balanced_accuracy'] for m in repeat_metrics[sex][model]])
                metrics[sex][model]['roc_aucs'][repeat] = np.mean(
                    [m['roc_auc'] for m in repeat_metrics[sex][model]])
                metrics[sex][model]['pr_aucs'][repeat] = np.mean(
                    [m['pr_auc'] for m in repeat_metrics[sex][model]])

        if (repeat + 1) % 10 == 0:
            print(f"Completed {repeat+1}/{N_REPEATS} repeats")

    return metrics

def evaluate_models_with_ensemble(X_train, X_test, y_train, y_test):
    """Evaluate all models including ensemble on current fold"""
    metrics = {model: {} for model in MODELS + ['ensemble']}
    
    # Train base models first
    base_models = {}
    base_predictions = {}
    
    for model in MODELS:
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegressionCV(
                Cs=C_VALUES, penalty='l2', cv=N_SPLITS, 
                max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE))
        ])
        
        # Train model
        pipeline.fit(X_train[model], y_train)
        base_models[model] = pipeline
        
        # Get predictions
        y_pred = pipeline.predict(X_test[model])
        y_prob = pipeline.predict_proba(X_test[model])[:, 1]
        base_predictions[model] = y_prob
        
        # Calculate metrics
        metrics[model]['accuracy'] = accuracy_score(y_test, y_pred)
        metrics[model]['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        metrics[model]['roc_auc'] = roc_auc_score(y_test, y_prob)
        metrics[model]['pr_auc'] = average_precision_score(y_test, y_prob)
    
    # Generate out-of-fold predictions for training ensemble
    oof_preds = {}
    for model in MODELS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds[model] = cross_val_predict(
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegressionCV(Cs=C_VALUES, penalty='l2', cv=N_SPLITS, 
                                           max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE))
            ]),
            X_train[model], y_train, method='predict_proba', cv=skf, n_jobs=-1
        )[:, 1]
    
    # Train ensemble model
    ensemble_train = np.column_stack([oof_preds[model] for model in MODELS])
    ensemble_test = np.column_stack([base_predictions[model] for model in MODELS])
    
    ensemble_model = LogisticRegressionCV(Cs=C_VALUES, penalty='l2', cv=N_SPLITS,
                                        max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
    ensemble_model.fit(ensemble_train, y_train)
    
    # Get ensemble predictions
    ensemble_pred = ensemble_model.predict(ensemble_test)
    ensemble_prob = ensemble_model.predict_proba(ensemble_test)[:, 1]
    
    # Add ensemble metrics
    metrics['ensemble'] = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, ensemble_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_prob),
        'pr_auc': average_precision_score(y_test, ensemble_prob)
    }
    
    return metrics

def save_results(metrics, control_only):
    """Save sex-specific metrics"""
    file_suffix = '_control' if control_only else '_control_moderate'
    
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    for model in MODELS + ['ensemble']:
        os.makedirs(f'results/{model}', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    for sex in ['male', 'female']:
        for model in MODELS + ['ensemble']:
            np.save(f'results/{model}/logreg_{model}{file_suffix}_{sex}_accuracies.npy',
                    metrics[sex][model]['accuracies'])
            np.save(f'results/{model}/logreg_{model}{file_suffix}_{sex}_balanced_accuracies.npy',
                    metrics[sex][model]['balanced_accuracies'])
            np.save(f'results/{model}/logreg_{model}{file_suffix}_{sex}_roc_aucs.npy',
                    metrics[sex][model]['roc_aucs'])
            np.save(f'results/{model}/logreg_{model}{file_suffix}_{sex}_pr_aucs.npy',
                    metrics[sex][model]['pr_aucs'])

def generate_report(metrics, control_only):
    """Generate comparative performance report"""
    report = ["Sex-Specific Performance Report\n" + "="*50 + "\n"]
    
    for model in MODELS + ['ensemble']:
        male_roc = np.mean(metrics['male'][model]['roc_aucs'])
        female_roc = np.mean(metrics['female'][model]['roc_aucs'])
        male_bal_acc = np.mean(metrics['male'][model]['balanced_accuracies'])
        female_bal_acc = np.mean(metrics['female'][model]['balanced_accuracies'])
        male_pr_auc = np.mean(metrics['male'][model]['pr_aucs']) 
        female_pr_auc = np.mean(metrics['female'][model]['pr_aucs'])
        
        report.append(
            f"{model}:\n"
            f"  Male ROC AUC: {male_roc:.3f} ± {np.std(metrics['male'][model]['roc_aucs']):.3f}\n"
            f"  Female ROC AUC: {female_roc:.3f} ± {np.std(metrics['female'][model]['roc_aucs']):.3f}\n"
            f"  ROC AUC Difference (M-F): {male_roc - female_roc:.3f}\n"
            f"  Male PR AUC: {male_pr_auc:.3f} ± {np.std(metrics['male'][model]['pr_aucs']):.3f}\n"
            f"  Female PR AUC: {female_pr_auc:.3f} ± {np.std(metrics['female'][model]['pr_aucs']):.3f}\n"
            f"  PR AUC Difference (M-F): {male_pr_auc - female_pr_auc:.3f}\n"
            f"  Male Balanced Accuracy: {male_bal_acc:.3f} ± {np.std(metrics['male'][model]['balanced_accuracies']):.3f}\n"
            f"  Female Balanced Accuracy: {female_bal_acc:.3f} ± {np.std(metrics['female'][model]['balanced_accuracies']):.3f}\n"
            f"  Balanced Accuracy Difference (M-F): {male_bal_acc - female_bal_acc:.3f}\n"
        )
    
    report.append("\nSummary of Sex Differences (Male - Female):\n" + "-"*50)
    report.append(f"{'Model':<15} {'ROC AUC':<12} {'PR AUC':<12} {'Bal. Acc.':<12}")
    report.append("-"*50)
    
    for model in MODELS + ['ensemble']:
        roc_diff = np.mean(metrics['male'][model]['roc_aucs']) - np.mean(metrics['female'][model]['roc_aucs'])
        pr_diff = np.mean(metrics['male'][model]['pr_aucs']) - np.mean(metrics['female'][model]['pr_aucs'])
        bal_diff = np.mean(metrics['male'][model]['balanced_accuracies']) - np.mean(metrics['female'][model]['balanced_accuracies'])
        
        report.append(f"{model:<15} {roc_diff:+.3f}        {pr_diff:+.3f}        {bal_diff:+.3f}")
    
    file_suffix = '_control' if control_only else '_control_moderate'
    with open(f'results/reports/sex_comparison_report{file_suffix}.txt', 'w') as f:
        f.write("\n".join(report))
    
    # Also print summary to console
    print("\nSummary of Sex Differences (Male - Female):")
    print("-"*50)
    print(f"{'Model':<15} {'ROC AUC':<12} {'PR AUC':<12} {'Bal. Acc.':<12}")
    print("-"*50)
    
    for model in MODELS + ['ensemble']:
        roc_diff = np.mean(metrics['male'][model]['roc_aucs']) - np.mean(metrics['female'][model]['roc_aucs'])
        pr_diff = np.mean(metrics['male'][model]['pr_aucs']) - np.mean(metrics['female'][model]['pr_aucs'])
        bal_diff = np.mean(metrics['male'][model]['balanced_accuracies']) - np.mean(metrics['female'][model]['balanced_accuracies'])
        
        print(f"{model:<15} {roc_diff:+.3f}        {pr_diff:+.3f}        {bal_diff:+.3f}")

if __name__ == "__main__":
    # Configuration
    N_SPLITS = 5
    N_REPEATS = 100
    RANDOM_STATE = 42
    MODELS = ['SC', 'FC', 'demos']  # Can also add FCgsr
    C_VALUES = np.logspace(-4, 4, 15)
    SAVE_RESULTS = False
    CONTROL_ONLY = False  # Set to True for control-only analysis

    np.random.seed(RANDOM_STATE) 
    random_ints = np.random.randint(0, 1000, N_REPEATS + N_SPLITS) 
    
    print("Loading data...")
    X_dict, y, sex_data = load_data(control_only=CONTROL_ONLY)
    site_data = np.load(f'data/training_data/aligned/site_location_control_moderate.npy')

    print(f"Data shapes: { {m: X_dict[m].shape for m in MODELS} }")
    print(f"Class balance: {np.mean(y):.2f} positive class\n")
    
    print("Starting sex-specific evaluation...")
    start_time = time.time()
    metrics = run_sex_specific_evaluation(X_dict, y, sex_data, site_data, random_ints)
    print(f"\nCompleted in {time.time()-start_time:.1f} seconds")
    
    if SAVE_RESULTS:
        print("\nSaving results...")
        save_results(metrics, CONTROL_ONLY)
        
    print("Generating report...")
    generate_report(metrics, CONTROL_ONLY)
    
    print("\nFinal Results:")
    generate_report(metrics, CONTROL_ONLY)