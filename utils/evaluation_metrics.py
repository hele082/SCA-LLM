import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import einops
from typing import Union



def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def calculate_spectral_efficiency(h_est: torch.Tensor,
                                  h_true: torch.Tensor,
                                  Nr: int,
                                  Nt: int,
                                  snr_db: float,
                                  device: torch.device,
                                  average_over_pred_len: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates Spectral Efficiency (SE) based on estimated and true channels.
    Assumes a standard MIMO setup with matched filtering at the receiver.

    Args:
        h_est (torch.Tensor): Estimated channel tensor (B, pred_len, Nr*Nt*2) (real tensor representing complex).
        h_true (torch.Tensor): True channel tensor (B, pred_len, Nr*Nt*2) (real tensor representing complex).
        Nr (int): Number of receive antennas.
        Nt (int): Number of transmit antennas.
        snr_db (float): Signal-to-Noise Ratio in dB.
        device (torch.device): Computation device ('cpu', 'cuda:0', etc.).
        average_over_pred_len (bool, optional): Whether to average the SE over the pred_len dimension. 
                                                If True, returns scalar tensors. 
                                                If False, returns tensors of shape (pred_len,). Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - se_est (torch.Tensor): SE based on estimated channel (scalar or shape (pred_len,)), in bps/Hz.
            - se_true (torch.Tensor): SE based on true channel (optimal matched filter) (scalar or shape (pred_len,)), in bps/Hz.
    """
    # --- Input Validation and Setup ---
    if h_est.shape != h_true.shape:
        raise ValueError(f"Shapes of h_est {h_est.shape} and h_true {h_true.shape} must match.")
    if len(h_est.shape) != 3:
         raise ValueError(f"Input tensors must have 3 dimensions (B, pred_len, feature_size), but got {len(h_est.shape)}.")
    if Nr <= 0 or Nt <= 0: # Handle invalid dimensions
        raise ValueError(f"Nr ({Nr}) and Nt ({Nt}) must be positive.")
         
    B, pred_len, feature_size = h_est.shape
    expected_feature_size = Nr * Nt * 2
    if feature_size != expected_feature_size:
        raise ValueError(f"Feature size ({feature_size}) does not match Nr*Nt*2 ({expected_feature_size}).")
        

    # --- Reshape real tensor to complex matrix ---
    # (B, pred_len, Nr*Nt*2) -> (B, pred_len, Nr, Nt, 2) -> (B, pred_len, Nr, Nt) complex
    try:
        h_est_reshaped = einops.rearrange(h_est, 'b p (nr nt c) -> b p nr nt c', nr=Nr, nt=Nt, c=2)
        H_est = torch.view_as_complex(h_est_reshaped.contiguous()).to(device) # (B, pred_len, Nr, Nt)

        h_true_reshaped = einops.rearrange(h_true, 'b p (nr nt c) -> b p nr nt c', nr=Nr, nt=Nt, c=2)
        H_true = torch.view_as_complex(h_true_reshaped.contiguous()).to(device) # (B, pred_len, Nr, Nt)
    except Exception as e:
        raise ValueError(f"Error reshaping tensors with einops: {e}. Check Nr, Nt and input shape.")


    # --- Signal Covariance (Assume Identity) ---
    # Expand S to match (B, pred_len, Nt, Nt)
    S = torch.eye(Nt, dtype=torch.complex64, device=device).unsqueeze(0).unsqueeze(0).expand(B, pred_len, -1, -1)

    # --- Noise Variance Calculation ---
    # Calculate noise variance based on the true channel H_true and SNR
    # H_true shape: (B, pred_len, Nr, Nt)
    signal_power_term = H_true # Since S=I, H_true @ S = H_true
    # Calculate squared Frobenius norm over Nr and Nt dimensions
    fro_sq = torch.linalg.norm(signal_power_term, ord='fro', dim=(2, 3)).pow(2) # Shape: (B, pred_len)

    # Average signal power per receive dimension (Nr * Nt total dimensions)
    avg_signal_power_per_dim = fro_sq / (Nr * Nt) # Shape: (B, pred_len)

    # Calculate noise variance per dimension
    noise_var_scalar = avg_signal_power_per_dim * (10.0 ** (-snr_db / 10.0)) # Shape: (B, pred_len)

    # Expand noise_var for broadcasting: (B, pred_len) -> (B, pred_len, 1, 1)
    noise_var = noise_var_scalar.unsqueeze(-1).unsqueeze(-1)

    # Clamp noise variance to avoid numerical issues
    noise_var = torch.clamp(noise_var, min=1e-20)

    # --- Matched Filter (MF) Receiver Design ---
    # Shapes are now (B, pred_len, ...)
    # 1. Based on Estimated Channel H_est
    D_est = torch.adjoint(H_est) # Shape: (B, pred_len, Nt, Nr)
    norm_D_est = torch.linalg.norm(D_est, ord='fro', dim=(2, 3), keepdim=True) # Shape: (B, pred_len, 1, 1)
    norm_D_est = torch.clamp(norm_D_est, min=1e-10) # Avoid division by zero
    D_est_normalized = D_est / norm_D_est # Shape: (B, pred_len, Nt, Nr)

    # 2. Based on True Channel H_true (Optimal MF)
    D_true = torch.adjoint(H_true) # Shape: (B, pred_len, Nt, Nr)
    norm_D_true = torch.linalg.norm(D_true, ord='fro', dim=(2, 3), keepdim=True) # Shape: (B, pred_len, 1, 1)
    norm_D_true = torch.clamp(norm_D_true, min=1e-10)
    D_true_normalized = D_true / norm_D_true # Shape: (B, pred_len, Nt, Nr)

    # --- Effective Channel Calculation ---
    # Shapes are now (B, pred_len, ...)
    # Effective channel: D_normalized @ H_true
    effective_channel_est = torch.matmul(D_est_normalized, H_true) # Shape: (B, pred_len, Nt, Nt)
    effective_channel_true = torch.matmul(D_true_normalized, H_true) # Shape: (B, pred_len, Nt, Nt)

    # --- SE Calculation (Using H_eff @ H_eff^H formulation) ---
    # Shapes are now (B, pred_len, ...)
    # SE = log2( det( S + (1/noise_var) * H_eff @ H_eff^H ) ) where S=I

    term_est = torch.matmul(effective_channel_est, torch.adjoint(effective_channel_est)) # Shape: (B, pred_len, Nt, Nt)
    term_true = torch.matmul(effective_channel_true, torch.adjoint(effective_channel_true)) # Shape: (B, pred_len, Nt, Nt)

    # Term inside determinant
    term_inside_det_est = S + torch.div(term_est, noise_var) # Shape: (B, pred_len, Nt, Nt)
    term_inside_det_true = S + torch.div(term_true, noise_var) # Shape: (B, pred_len, Nt, Nt)

    # Calculate determinant and then log2
    det_est = torch.linalg.det(term_inside_det_est) # Shape: (B, pred_len)
    # Clamp absolute value before log2 to avoid log(0) or log(negative) due to numerical precision
    se_est_per_sample = torch.log2(torch.clamp(torch.abs(det_est), min=1e-12)) # Shape: (B, pred_len)

    det_true = torch.linalg.det(term_inside_det_true) # Shape: (B, pred_len)
    se_true_per_sample = torch.log2(torch.clamp(torch.abs(det_true), min=1e-12)) # Shape: (B, pred_len)

    # --- Averaging ---
    # Average over the batch dimension first
    se_est_avg_batch = torch.mean(se_est_per_sample.real, dim=0) # Shape: (pred_len,)
    se_true_avg_batch = torch.mean(se_true_per_sample.real, dim=0) # Shape: (pred_len,)

    # Optionally average over the prediction length dimension
    if average_over_pred_len:
        se_est_final = torch.mean(se_est_avg_batch) # Scalar tensor
        se_true_final = torch.mean(se_true_avg_batch) # Scalar tensor
    else:
        se_est_final = se_est_avg_batch # Shape: (pred_len,)
        se_true_final = se_true_avg_batch # Shape: (pred_len,)

    return se_est_final, se_true_final 


def calculate_nmse(h_est: torch.Tensor, h_true: torch.Tensor, average_over_pred_len: bool = True) -> torch.Tensor:
    """
    Calculates Normalized Mean Squared Error (NMSE) between estimated and true tensors.
    NMSE = ||h_est - h_true||^2 / ||h_true||^2

    Args:
        h_est (torch.Tensor): Estimated tensor (B, pred_len, Features).
        h_true (torch.Tensor): True tensor (B, pred_len, Features).
        average_over_pred_len (bool): If True, average NMSE across the pred_len dimension.

    Returns:
        torch.Tensor: Calculated NMSE (scalar if average_over_pred_len=True, else shape (pred_len,)).
    """
    if h_est.shape != h_true.shape:
        raise ValueError(f"Shapes of h_est {h_est.shape} and h_true {h_true.shape} must match.")
    if len(h_est.shape) != 3:
         raise ValueError(f"Input tensors must have 3 dimensions (B, pred_len, feature_size), but got {len(h_est.shape)}.")

    # Calculate squared Frobenius norm for error and true signal per sample
    # Sum over the last dimension (features)
    error_power = torch.sum((h_est - h_true) ** 2, dim=-1) # Shape: (B, pred_len)
    true_power = torch.sum(h_true ** 2, dim=-1)           # Shape: (B, pred_len)

    # Avoid division by zero for samples where true power is very small
    true_power = torch.clamp(true_power, min=1e-10)

    # NMSE per sample
    nmse_per_sample = error_power / true_power # Shape: (B, pred_len)

    # Average over batch dimension
    nmse_avg_batch = torch.mean(nmse_per_sample, dim=0) # Shape: (pred_len,)

    # Optionally average over prediction length
    if average_over_pred_len:
        nmse_final = torch.mean(nmse_avg_batch) # Scalar
    else:
        nmse_final = nmse_avg_batch # Shape: (pred_len,)

    return nmse_final

# Alias for clarity when using it for per-step calculation
# This avoids code duplication and makes the intent clear in the calling function.
calculate_nmse_per_step = lambda h_est, h_true: calculate_nmse(h_est, h_true, average_over_pred_len=False) 

# Alias for spectral efficiency per step calculation
calculate_spectral_efficiency_per_step = lambda h_est, h_true, Nr, Nt, snr_db, device: calculate_spectral_efficiency(
    h_est, h_true, Nr, Nt, snr_db, device, average_over_pred_len=False
) 

# Verification function for multi-model comparison
def verify_true_se_consistency(model_results, tolerance=1e-3):
    """
    Verifies that true SE values are consistent across different models
    for the same (velocity, SNR) conditions.
    
    Args:
        model_results: Dictionary mapping model_name to result dictionary
        tolerance: Numerical tolerance for comparison
        
    Returns:
        bool: True if all true SE values are consistent, False otherwise
        dict: Inconsistencies found {(v,s): {model1: val1, model2: val2, ...}}
    """
    if len(model_results) <= 1:
        return True, {}  # No comparison needed for single model
        
    # Extract true SE values for each model and condition
    true_se_values = {}
    for model_name, results in model_results.items():
        for (v, s), value in results['se']['true']['avg'].items():
            if (v, s) not in true_se_values:
                true_se_values[(v, s)] = {}
            true_se_values[(v, s)][model_name] = value
    
    # Check consistency for each condition
    inconsistencies = {}
    for condition, model_values in true_se_values.items():
        if len(model_values) > 1:
            # Get values as list
            values = list(model_values.values())
            # Compare first value with all others
            reference = values[0]
            for model_name, value in model_values.items():
                if abs(value - reference) > tolerance:
                    # Inconsistency found
                    if condition not in inconsistencies:
                        inconsistencies[condition] = model_values
    
    return len(inconsistencies) == 0, inconsistencies 

def consolidate_true_se_for_comparison(model_results):
    """
    Consolidates true SE values across different models to ensure consistent
    reference values for comparison.
    
    Args:
        model_results: Dictionary mapping model_name to result dictionary
        
    Returns:
        dict: Consolidated true SE values {(v,s): value} from the first model
        dict: Updated model_results with consistent true SE values
    """
    if len(model_results) <= 1:
        return {}, model_results  # No consolidation needed
    
    # Get first model's true SE values as reference
    first_model = next(iter(model_results))
    reference_true_se = model_results[first_model]['se']['true']['avg']
    
    # Update all models to use the same true SE values
    updated_results = {}
    for model_name, results in model_results.items():
        # Create a deep copy to avoid modifying original
        model_copy = results.copy()
        model_copy['se'] = model_copy['se'].copy()
        model_copy['se']['true'] = model_copy['se']['true'].copy()
        
        # Replace true SE values with reference values
        model_copy['se']['true']['avg'] = reference_true_se
        updated_results[model_name] = model_copy
    
    return reference_true_se, updated_results 