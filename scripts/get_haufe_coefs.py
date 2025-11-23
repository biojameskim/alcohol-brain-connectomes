import numpy as np
from sklearn.preprocessing import StandardScaler

def get_haufe_coefs(matrix_type, file_name, sex):
    """
    [get_haufe_coefs] calculates the normalized Haufe coefficients for a given set of coefficients and input data.
    First the average coefficients are calculated (over the number of iterations and then transformed and normalized)
    Normalization: Coefficients are divided by the max absolute value to put them on a [0,1] scale.
    This allows for visualization without scale distraction caused by different ridge penalties across models.
    Source: (Haufe et. al 2014) https://www.sciencedirect.com/science/article/pii/S1053811913010914#s0210
    """
    coefs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_coefficients.npy')
    X = np.load(f'data/training_data/aligned/X_{matrix_type}_{file_name}{sex}.npy')
    X = StandardScaler().fit_transform(X) # Standardize the data

    coefs = np.mean(coefs, axis=0) # Average the coefficients over the number of iterations
    cov_X = np.cov(X, rowvar=False) # Covariance matrix of X (cols are features so rowvar=False)
                                                    
    haufe = cov_X @ coefs
    
    # Normalize by max absolute value (each model independently)
    haufe_normalized = haufe / np.max(np.abs(haufe))
    
    return haufe_normalized

def get_haufe_coefs_unnormalized(matrix_type, file_name, sex):
    """
    [get_haufe_coefs_unnormalized] calculates the Haufe coefficients for a given set of coefficients and input data.
    First the average coefficients are calculated (over the number of iterations and then transformed)
    Source: (Haufe et. al 2014) https://www.sciencedirect.com/science/article/pii/S1053811913010914#s0210
    """
    coefs = np.load(f'results/{matrix_type}/logreg_{matrix_type}_{file_name}{sex}_coefficients.npy')
    X = np.load(f'data/training_data/aligned/X_{matrix_type}_{file_name}{sex}.npy')
    X = StandardScaler().fit_transform(X) # Standardize the data

    coefs = np.mean(coefs, axis=0) # Average the coefficients over the number of iterations
    cov_X = np.cov(X, rowvar=False) # Covariance matrix of X (cols are features so rowvar=False)
    return cov_X @ coefs