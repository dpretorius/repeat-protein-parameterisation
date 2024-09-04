import numpy as np
from helix_fitting import cost_function_2
from secondaryhelix_fitting import cost_function_3


def approximate_hessian_from_gradients(grad, x_values, epsilon=1e-5):
    n = len(x_values)
    H = np.zeros((n, n))
    for i in range(n):
        x1 = x_values.copy()
        x2 = x_values.copy()
        x1[i] += epsilon
        x2[i] -= epsilon
        
        g1 = grad(x1)
        g2 = grad(x2)
        
        H[:, i] = (g1 - g2) / (2 * epsilon)
    return H

def calculate_uncertainties_from_grad(grad, x_values, fun_value, n_params, epsilon=1e-5):
    # Approximate the Hessian matrix from gradient vectors
    H = approximate_hessian_from_gradients(grad, x_values, epsilon)
    
    # Compute the inverse of the Hessian to get the covariance matrix
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)
    
    cov = H_inv * fun_value / n_params
    
    # Make a copy of the diagonal elements to modify them
    diag_cov = np.copy(np.diag(cov))
    diag_cov[diag_cov < 0] = 0  # Replace negative values with zero
    
    # Calculate uncertainties
    uncertainties = np.sqrt(diag_cov)
    
    return uncertainties, cov

def propagate_uncertainty(cov, jac):
    if jac.ndim == 1:
        jac = jac.reshape(1, -1)
    propagated_cov = jac @ cov @ jac.T
    propagated_cov[propagated_cov < 0] = 0  # Ensure no negative values
    return np.sqrt(propagated_cov.item())

def numerical_jacobian(func, x_values, epsilon=1e-5):
    n = len(x_values)
    jac = np.zeros(n)
    f0 = func(x_values)
    for i in range(n):
        x1 = x_values.copy()
        x1[i] += epsilon
        jac[i] = (func(x1) - f0) / epsilon
    return jac

def gradient_function(params, centroids, handedness, epsilon=1e-5):
    n = len(params)
    grad = np.zeros(n)
    for i in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        grad[i] = (cost_function_2(params_plus, centroids, handedness) - 
                   cost_function_2(params_minus, centroids, handedness)) / (2 * epsilon)
    return grad

def gradient_function_3(params, Ca_residues, test_helix, t_arr, B, N, epsilon=1e-5):
    n = len(params)
    grad = np.zeros(n)
    for i in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        grad[i] = (cost_function_3(params_plus, Ca_residues, test_helix, t_arr, B, N) - 
                   cost_function_3(params_minus, Ca_residues, test_helix, t_arr, B, N)) / (2 * epsilon)
    return grad
