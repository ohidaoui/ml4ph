"""
Functions for making predictions using fitted Bayesian linear regression model
"""

import numpy as np
import pandas as pd
from scipy import stats
import arviz as az
import matplotlib.pyplot as plt

def plot_forest_with_features(idata, feature_names, var_names=['beta'], combined=True, hdi_prob=0.95):
    """
    Create a forest plot with feature names instead of parameter indices.
    """
    # Create the forest plot
    ax = az.plot_forest(idata, 
                       var_names=var_names,
                       combined=combined,
                       hdi_prob=hdi_prob,
                       show=False)
    
    # Get the current y-axis labels
    current_labels = ax[0].get_yticklabels()
    
    # Create new labels with feature names
    new_labels = []
    for label in current_labels:
        # Extract the parameter index from the label
        param_idx = int(label.get_text().split('[')[1].split(']')[0])
        # Use the corresponding feature name
        new_labels.append(feature_names[param_idx])
    
    # Set the new labels
    ax[0].set_yticklabels(new_labels)
    
    plt.tight_layout()
    return ax

def get_posterior_samples(fit):
    """
    Extract posterior samples of parameters from the fitted model.
    """
    # Extract posterior samples
    posterior_samples = fit.draws_pd()
    
    # Get parameter names
    K = fit.stan_variable("beta").shape[1]
    alpha_samples = posterior_samples['alpha'].values
    beta_samples = posterior_samples[[f'beta[{i+1}]' for i in range(K)]].values
    sigma_samples = posterior_samples['sigma'].values
    
    return {
        'alpha': alpha_samples,
        'beta': beta_samples,
        'sigma': sigma_samples
    }
    

def predict_mean(X_new, posterior_params):
    """
    Predict mean life expectancy for new data points using posterior samples.
    """
    # Extract parameters
    alpha = posterior_params['alpha']
    beta = posterior_params['beta']
    
    # Calculate predictions for each posterior sample
    predictions = []
    for i in range(len(alpha)):
        y_pred = alpha[i] + X_new @ beta[i].T
        predictions.append(y_pred)
    
    # Convert to array
    predictions = np.array(predictions)
    
    # Calculate mean and credible intervals
    mean_pred = predictions.mean(axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    
    return mean_pred, (ci_lower, ci_upper)

def predict_distribution(X_new, posterior_params, n_samples=1000):
    """
    Generate samples from the posterior predictive distribution.
    """
    # Extract parameters
    alpha = posterior_params['alpha']
    beta = posterior_params['beta']
    sigma = posterior_params['sigma']
    
    # Randomly select posterior samples
    idx = np.random.randint(0, len(alpha), n_samples)
    alpha_samples = alpha[idx]
    beta_samples = beta[idx]
    sigma_samples = sigma[idx]
    
    # Generate predictions with uncertainty
    predictions = []
    for i in range(n_samples):
        # Calculate mean prediction
        mean_pred = alpha_samples[i] + X_new @ beta_samples[i].T
        
        # Add noise
        y_pred = np.random.normal(mean_pred, sigma_samples[i])
        predictions.append(y_pred)
    
    return np.array(predictions)

def get_posterior_samples_longitudinal(fit):
    """
    Extract posterior samples of parameters from the fitted longitudinal model.
    """
    # Extract posterior samples
    posterior_samples = fit.draws_pd()

    # Extract scalar parameters
    alpha_samples = posterior_samples['alpha'].values
    gamma_samples = posterior_samples['gamma'].values
    sigma_samples = posterior_samples['sigma'].values
    sigma_u_samples = posterior_samples['sigma_u'].values

    # Extract vector parameters (matrix)
    beta_cols = [col for col in posterior_samples.columns if col.startswith('beta[')]
    beta_samples = posterior_samples[beta_cols].values
    u_cols = [col for col in posterior_samples.columns if col.startswith('u[')]
    u_samples = posterior_samples[u_cols].values

    return {
        'alpha': alpha_samples,
        'gamma': gamma_samples,
        'beta': beta_samples,
        'u': u_samples,
        'sigma': sigma_samples,
        'sigma_u': sigma_u_samples
    }
    

def predict_longitudinal(X_new, country_indices, years, posterior_params, n_samples=1000):
    """
    Predict mean life expectancy for new data points using posterior samples.
    """
    # Extract parameters
    alpha = posterior_params['alpha']
    gamma = posterior_params['gamma']
    beta = posterior_params['beta']
    u = posterior_params['u']
    sigma = posterior_params['sigma']
    sigma_u = posterior_params['sigma_u']
    
    # Randomly select posterior samples
    idxs = np.random.randint(0, len(alpha), n_samples)

    # Calculate predictions for each posterior sample
    predictions = []
    for i in idxs:
        mu_pred = alpha[i] + u[i][country_indices-1] + gamma[i] * years + X_new @ beta[i].T
        y_pred = np.random.normal(mu_pred, sigma[i])
        predictions.append(y_pred)
    
    # Convert to array
    predictions = np.array(predictions)
    
    # Calculate mean and credible intervals
    mean_pred = predictions.mean(axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    
    return predictions, mean_pred, (ci_lower, ci_upper)


def summarize_predictions(predictions, actual=None):
    """
    Summarize predictions with uncertainty metrics.
    """
    # Calculate summary statistics
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'Mean_Prediction': mean_pred,
        'Std_Prediction': std_pred,
        'CI_2.5%': ci_lower,
        'CI_97.5%': ci_upper
    })
    
    if actual is not None:
        # Add actual values and calculate error metrics
        summary['Actual'] = actual
        summary['Error'] = actual - mean_pred
        summary['Abs_Error'] = abs(summary['Error'])
        summary['In_CI'] = (actual >= ci_lower) & (actual <= ci_upper)
        
        # Calculate coverage probability
        coverage_prob = summary['In_CI'].mean()
        print(f"95% CI Coverage: {coverage_prob:.2%}")
    
    return summary
