import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from .evaluation_metrics import verify_true_se_consistency, consolidate_true_se_for_comparison

def load_results(result_path):
    """
    Load results from either pickle or JSON format.
    
    Args:
        result_path: Path to the results file (.pkl or .json)
        
    Returns:
        dict: Loaded results dictionary
    """
    if result_path.endswith('.pkl'):
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
            if 'results' in data:
                return data['results']
            return data
    elif result_path.endswith('.json'):
        with open(result_path, 'r') as f:
            data = json.load(f)
            
            # Convert JSON format back to internal format
            if 'nmse' in data and 'se' in data:
                # Convert string keys back to tuples
                results = {
                    'nmse': {
                        'avg': {},
                        'per_step': {}
                    },
                    'se': {
                        'est': {
                            'avg': {},
                            'per_step': {}
                        },
                        'true': {
                            'avg': {},
                            'per_step': {}
                        }
                    }
                }
                
                # Process NMSE data
                for key, value in data['nmse']['average'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['nmse']['avg'][(int(v), float(s))] = value
                
                for key, values in data['nmse']['per_step'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['nmse']['per_step'][(int(v), float(s))] = values
                
                # Process SE data
                for key, value in data['se']['estimated']['average'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['se']['est']['avg'][(int(v), float(s))] = value
                    
                for key, values in data['se']['estimated']['per_step'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['se']['est']['per_step'][(int(v), float(s))] = values
                    
                for key, value in data['se']['true']['average'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['se']['true']['avg'][(int(v), float(s))] = value
                    
                for key, values in data['se']['true']['per_step'].items():
                    v, s = key.split('km/h_')[0], key.split('km/h_')[1].replace('dB', '')
                    results['se']['true']['per_step'][(int(v), float(s))] = values
                
                return results
            
            return data
    else:
        raise ValueError(f"Unsupported file format: {result_path}")

def load_multiple_model_results(result_files):
    """
    Load multiple model results from a dictionary of {model_name: file_path}.
    
    Args:
        result_files: Dictionary mapping model names to result file paths
        
    Returns:
        dict: Dictionary mapping model names to result dictionaries
    """
    model_results = {}
    for model_name, file_path in result_files.items():
        try:
            model_results[model_name] = load_results(file_path)
        except Exception as e:
            print(f"Error loading results for {model_name}: {e}")
    
    return model_results

def plot_se_comparison(model_results, 
                       velocities, 
                       snr_value=10, 
                       save_path=None, 
                       show_plot=True):
    """
    Plot spectral efficiency comparison for multiple models.
    Ensures true SE is consistent across all models.
    
    Args:
        model_results: Dictionary mapping model_name to result dictionary
        velocities: List of velocity values to plot
        snr_value: SNR value to plot at (fixed SNR)
        save_path: Path to save the plot (None = don't save)
        show_plot: Whether to display the plot
    """
    # Verify true SE consistency
    is_consistent, inconsistencies = verify_true_se_consistency(model_results)
    if not is_consistent:
        print("Warning: True SE values inconsistent across models!")
        for cond, values in inconsistencies.items():
            print(f"  Condition {cond}: {values}")
    
    # Consolidate true SE values
    reference_true_se, updated_results = consolidate_true_se_for_comparison(model_results)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot estimated SE for each model
    for model_name, results in updated_results.items():
        se_values = []
        for v in velocities:
            if (v, snr_value) in results['se']['est']['avg']:
                se_values.append(results['se']['est']['avg'][(v, snr_value)])
            else:
                se_values.append(np.nan)  # Handle missing values
        
        plt.plot(velocities, se_values, marker='o', label=f"{model_name}")
    
    # Plot true SE (theoretical upper bound) once
    true_se_values = []
    for v in velocities:
        if (v, snr_value) in reference_true_se:
            true_se_values.append(reference_true_se[(v, snr_value)])
        else:
            true_se_values.append(np.nan)
    
    plt.plot(velocities, true_se_values, 'k--', linewidth=2, label="True SE (Upper Bound)")
    
    # Customize plot
    plt.xlabel("Velocity (km/h)")
    plt.ylabel("Spectral Efficiency (bps/Hz)")
    plt.title(f"Spectral Efficiency Comparison at SNR = {snr_value} dB")
    plt.grid(True)
    plt.legend()
    
    # Save and/or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_se_vs_snr(model_results, 
                  velocity_value=30, 
                  snr_values=None, 
                  save_path=None, 
                  show_plot=True):
    """
    Plot spectral efficiency vs SNR for multiple models at fixed velocity.
    
    Args:
        model_results: Dictionary mapping model_name to result dictionary
        velocity_value: Fixed velocity to plot at
        snr_values: List of SNR values to plot (default: range 0-20 dB)
        save_path: Path to save the plot (None = don't save)
        show_plot: Whether to display the plot
    """
    if snr_values is None:
        snr_values = list(range(0, 21, 5))  # Default: 0, 5, 10, 15, 20 dB
    
    # Consolidate true SE values
    reference_true_se, updated_results = consolidate_true_se_for_comparison(model_results)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot estimated SE for each model
    for model_name, results in updated_results.items():
        se_values = []
        for s in snr_values:
            if (velocity_value, s) in results['se']['est']['avg']:
                se_values.append(results['se']['est']['avg'][(velocity_value, s)])
            else:
                se_values.append(np.nan)
        
        plt.plot(snr_values, se_values, marker='o', label=f"{model_name}")
    
    # Plot true SE (theoretical upper bound) once
    true_se_values = []
    for s in snr_values:
        if (velocity_value, s) in reference_true_se:
            true_se_values.append(reference_true_se[(velocity_value, s)])
        else:
            true_se_values.append(np.nan)
    
    plt.plot(snr_values, true_se_values, 'k--', linewidth=2, label="True SE (Upper Bound)")
    
    # Customize plot
    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Efficiency (bps/Hz)")
    plt.title(f"Spectral Efficiency vs SNR at Velocity = {velocity_value} km/h")
    plt.grid(True)
    plt.legend()
    
    # Save and/or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# Example usage
"""
Example of how to use these functions:

# Load results from different models
model_results = {
    "Model_A": load_results("path/to/modelA_results.pkl"),
    "Model_B": load_results("path/to/modelB_results.pkl"),
    "Model_C": load_results("path/to/modelC_results.pkl")
}

# Plot comparison
plot_se_comparison(model_results, 
                   velocities=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                   snr_value=10,
                   save_path="results/se_velocity_comparison.png")

plot_se_vs_snr(model_results,
              velocity_value=30,
              snr_values=[0, 5, 10, 15, 20],
              save_path="results/se_snr_comparison.png")
""" 