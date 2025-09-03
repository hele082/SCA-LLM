import os
import h5py
import torch
import numpy as np
import random
import gc
import time
from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Tuple, Optional, Dict, Callable
from einops import rearrange

# Import the metrics directly from utils
from utils.evaluation_metrics import (
    calculate_spectral_efficiency, 
    calculate_spectral_efficiency_per_step,
    calculate_nmse,
    calculate_nmse_per_step
)

# Global mapping functions for velocity and SNR
def velocity_idx_to_km_h(v_idx):
    """Convert velocity index to km/h value, assuming 5 km/h steps."""
    return v_idx * 5

def km_h_to_velocity_idx(velocity_km_h):
    """Convert km/h velocity to index, assuming 5 km/h steps."""
    return velocity_km_h // 5

# Default velocity values in km/h, from 0 to 60 km/h with 5 km/h steps
DEFAULT_VELOCITY_VALUES = list(range(0, 61, 5))  # 0, 5, 10, ..., 55, 60 km/h

# Default SNR values in dB, from 0 to 20 dB with 1 dB steps
DEFAULT_SNR_VALUES = list(range(0, 21, 1))  # 0, 1, 2, ..., 19, 20 dB

def snr_idx_to_db(snr_idx, snr_values=None):
    """
    Convert SNR index to dB value.
    
    Args:
        snr_idx: Index in the SNR array
        snr_values: List of SNR values in dB (uses DEFAULT_SNR_VALUES if None)
        
    Returns:
        SNR value in dB
    """
    snr_array = DEFAULT_SNR_VALUES if snr_values is None else snr_values
    if not 0 <= snr_idx < len(snr_array):
        raise ValueError(f"SNR index {snr_idx} out of range [0, {len(snr_array)-1}]")
    return snr_array[snr_idx]

def db_to_snr_idx(snr_db, snr_values=None):
    """
    Find the closest SNR index for a given dB value.
    
    Args:
        snr_db: SNR value in dB
        snr_values: List of SNR values in dB (uses DEFAULT_SNR_VALUES if None)
        
    Returns:
        Index of the closest SNR value
    """
    snr_array = DEFAULT_SNR_VALUES if snr_values is None else snr_values
    snr_array = np.array(snr_array)
    idx = np.abs(snr_array - snr_db).argmin()
    return idx

class FlexibleCSIDataset(Dataset):
    """
    A flexible CSI dataset implementation optimized for research experiments.
    
    Features:
    - Efficiently supports dynamic switching between velocity subsets
    - Allows flexible SNR selection for testing
    - Optimized for memory usage with lazy loading of subsets
    - Returns samples with shape (seq_len+pred_len, Nr*Nt*2)
    
    Data format: (Nv, Nu, T, Nsc, Nr, Nt) where:
    - Nv: Number of velocities (0-60km/h, step: 5km/h)
    - Nu: Number of users
    - T: Number of time slots
    - Nsc: Number of subcarriers
    - Nr: Number of receive antennas
    - Nt: Number of transmit antennas
    """
    def __init__(
        self,
        file_path: str,
        seq_len: int = 24,
        pred_len: int = 6,
        is_training: bool = True,
        snr_db: Union[float, List[float]] = 10.0,
        active_velocity_indices: Optional[List[int]] = None,
        user_subset: Optional[Union[int, List[int]]] = None,
        subcarrier_subset: Optional[Union[int, List[int]]] = None,
        verbose: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the HDF5 data file
            seq_len: Sequence length for history
            pred_len: Prediction length for future
            is_training: Whether in training mode (random SNR) or not
            snr_db: SNR value in dB or range [min, max] for random selection
            active_velocity_indices: List of velocity indices to use (None = all)
            user_subset: User indices to use (None = all)
            subcarrier_subset: Subcarrier indices to use (None = all)
            verbose: Whether to print verbose information
        """
        super(FlexibleCSIDataset, self).__init__()
        
        # Basic parameters
        self.file_path = file_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.is_training = is_training
        self.verbose = verbose
        self.window_len = seq_len + pred_len
        
        # SNR handling
        if isinstance(snr_db, list) and len(snr_db) == 2:
            self.snr_min, self.snr_max = snr_db
            self.current_snr = None  # Will be randomly selected in __getitem__
        else:
            self.snr_min = self.snr_max = float(snr_db)
            self.current_snr = float(snr_db)
        
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Load data info without loading all data
        self._load_data_info()
        
        # Set active velocity indices
        self.all_velocity_indices = list(range(self.Nv))
        self.active_velocity_indices = active_velocity_indices if active_velocity_indices is not None else self.all_velocity_indices
        self._validate_velocity_indices()
        
        # Set user and subcarrier subsets
        self.user_subset = self._parse_subset(user_subset, self.Nu, "user_subset")
        self.subcarrier_subset = self._parse_subset(subcarrier_subset, self.Nsc, "subcarrier_subset")
        
        # 始终预加载所有数据
        if self.verbose:
            print("Preloading all data into memory...")
        self._preload_all_data()
        
        # Generate sample indices
        self._generate_indices()
        
        if self.verbose:
            print(f"Dataset initialized with {len(self)} samples")
            print(f"Active velocities: {self.active_velocity_indices}")
            print(f"SNR range: {self.snr_min}-{self.snr_max} dB")
            print(f"Sequence length: {self.seq_len}, Prediction length: {self.pred_len}")
            
    def _load_data_info(self):
        """Load data dimensions without loading all data."""
        with h5py.File(self.file_path, 'r') as f:
            if 'channel_data' not in f:
                raise KeyError(f"Key 'channel_data' not found in file: {self.file_path}")
            
            # Just get shape information
            channel_data = f['channel_data']
            self.Nv = channel_data.shape[0]  # Number of velocities
            self.Nu = channel_data.shape[1]  # Number of users
            self.T = channel_data.shape[2]   # Number of time slots
            self.Nsc = channel_data.shape[3] # Number of subcarriers
            self.Nr = channel_data.shape[4]  # Number of receive antennas
            self.Nt = channel_data.shape[5]  # Number of transmit antennas
            self.data_dtype = channel_data.dtype
            
            if self.verbose:
                print(f"Data dimensions: Nv={self.Nv}, Nu={self.Nu}, T={self.T}, "
                      f"Nsc={self.Nsc}, Nr={self.Nr}, Nt={self.Nt}")
                print(f"Data dtype: {self.data_dtype}")
    
    def _parse_subset(self, subset_param, max_dim, name):
        """Parse subset parameters to indices."""
        if subset_param is None:
            return list(range(max_dim))
        elif isinstance(subset_param, int):
            if not 0 <= subset_param < max_dim:
                raise ValueError(f"Invalid {name} index: {subset_param} (max: {max_dim-1})")
            return [subset_param]
        elif isinstance(subset_param, list):
            if not all(0 <= i < max_dim for i in subset_param):
                raise ValueError(f"Invalid indices in {name}: {subset_param}")
            return sorted(list(set(subset_param)))  # Ensure unique and sorted
        else:
            raise TypeError(f"{name} must be None, int, or List[int]")
    
    def _validate_velocity_indices(self):
        """Validate velocity indices."""
        if not all(0 <= idx < self.Nv for idx in self.active_velocity_indices):
            invalid = [idx for idx in self.active_velocity_indices if not 0 <= idx < self.Nv]
            raise ValueError(f"Invalid velocity indices: {invalid}. Valid range: 0-{self.Nv-1}")
        
    def _preload_all_data(self):
        """Preload all data into memory, suitable for datasets around 1GB size"""
        with h5py.File(self.file_path, 'r') as f:
            # Load the entire dataset
            start_time = time.time()
            data_slice = f['channel_data'][:]
            
            # Convert to complex numbers
            real_part = data_slice['real'].astype(np.float32)
            imag_part = data_slice['imag'].astype(np.float32)
            complex_data = real_part + 1j * imag_part
            
            # Pre-reshape the data to avoid repeated rearrange operations in _load_data_window
            # Shape from (Nv, Nu, T, Nsc, Nr, Nt) to (Nv, Nu, T, Nsc, Nr*Nt)
            self.all_data = np.reshape(complex_data, 
                                       (self.Nv, self.Nu, self.T, self.Nsc, self.Nr*self.Nt))
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Data loading completed in {elapsed:.2f} seconds")
                print(f"Data shape: {self.all_data.shape}, Memory usage: {self.all_data.nbytes / (1024**3):.2f} GB")
                
    def _generate_indices(self):
        """Generate sample indices for the active velocities."""
        self.sample_indices = []
        
        for v_idx in self.active_velocity_indices:
            # Calculate number of windows per user and subcarrier
            num_windows = max(1, self.T // self.window_len)
            
            for u_idx in self.user_subset:
                for sc_idx in self.subcarrier_subset:
                    for w_idx in range(num_windows):
                        # Store index tuple: (velocity_idx, user_idx, subcarrier_idx, window_idx)
                        self.sample_indices.append((v_idx, u_idx, sc_idx, w_idx))
                
    def _load_data_window(self, v_idx, u_idx, sc_idx, w_idx):
        """
        Load a specific data window.
        
        Args:
            v_idx: Velocity index
            u_idx: User index
            sc_idx: Subcarrier index
            w_idx: Window index
            
        Returns:
            Complex numpy array of shape (window_len, Nr*Nt)
        """
        # Calculate window range
        start_t = w_idx * self.window_len
        end_t = start_t + self.window_len
        
        # Directly extract from preloaded data that's already reshaped
        # No need for rearrange operation as the reshaping was done during loading (too slow)
        return self.all_data[v_idx, u_idx, start_t:end_t, sc_idx]
    
    def set_active_velocities(self, velocity_indices):
        """
        Set active velocities for testing different velocity subsets.
        
        Args:
            velocity_indices: List of velocity indices to use (0-12 for 0-60km/h)
        """
        if velocity_indices is None:
            self.active_velocity_indices = self.all_velocity_indices
        else:
            self.active_velocity_indices = velocity_indices
            self._validate_velocity_indices()
        
        # Regenerate sample indices for new velocity subset
        self._generate_indices()
        
        if self.verbose:
            print(f"Active velocities updated to: {self.active_velocity_indices}")
            print(f"Dataset now has {len(self)} samples")
    
    def set_active_velocities_km_h(self, velocities_km_h: List[int]):
        """
        Set active velocities using km/h values (more intuitive than indices).
        
        Args:
            velocities_km_h: List of velocities in km/h to use (0-60km/h)
        """
        # Convert km/h values to indices
        velocity_indices = [km_h_to_velocity_idx(v) for v in velocities_km_h]
        
        # Use the existing method to set active velocities
        self.set_active_velocities(velocity_indices)
        
        if self.verbose:
            print(f"Active velocities set to: {velocities_km_h} km/h")
    
    def set_snr(self, snr_db):
        """
        Set SNR for testing.
        
        Args:
            snr_db: SNR value in dB or range [min, max] for random selection
        """
        if isinstance(snr_db, list) and len(snr_db) == 2:
            self.snr_min, self.snr_max = snr_db
            self.current_snr = None
        else:
            self.snr_min = self.snr_max = float(snr_db)
            self.current_snr = float(snr_db)
        
        if self.verbose:
            if self.snr_min == self.snr_max:
                print(f"SNR set to fixed value: {self.snr_min} dB")
            else:
                print(f"SNR range set to: {self.snr_min}-{self.snr_max} dB")
    
    def add_awgn(self, H, snr_db):
        """
        Add AWGN noise to complex channel data.
        
        Args:
            H: Complex channel data
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy channel data
        """
        # Calculate noise power based on SNR
        signal_power = np.mean(np.abs(H) ** 2)
        noise_power = signal_power * (10 ** (-snr_db / 10))
        
        # Generate complex Gaussian noise
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
        
        return H + noise
    
    def complex_to_real(self, H):
        """
        Convert complex data to real by concatenating real and imaginary parts.
        
        Args:
            H: Complex numpy array
            
        Returns:
            Real numpy array with doubled last dimension
        """
        return np.concatenate([np.real(H), np.imag(H)], axis=-1)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_indices)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with 'history' and 'future' tensors
        """
        # Get indices for this sample
        v_idx, u_idx, sc_idx, w_idx = self.sample_indices[index]
        
        # Load data window
        H_complex = self._load_data_window(v_idx, u_idx, sc_idx, w_idx)
        
        # Split into history and future
        H_his_clean = H_complex[:self.seq_len]
        H_future_clean = H_complex[self.seq_len:self.seq_len+self.pred_len]
        
        # Determine SNR
        if self.is_training or self.current_snr is None:
            snr = random.uniform(self.snr_min, self.snr_max)
        else:
            snr = self.current_snr
        
        # Add noise only to history part
        H_his_noisy = self.add_awgn(H_his_clean, snr)
        
        # Normalize using statistics from noisy history
        power = np.mean(np.abs(H_his_noisy) ** 2)
        norm_factor = np.sqrt(power)
        
        H_his_norm = H_his_noisy / norm_factor
        H_future_norm = H_future_clean / norm_factor
        
        # Convert to real representation
        H_his_real = self.complex_to_real(H_his_norm)
        H_future_real = self.complex_to_real(H_future_norm)
        
        # Convert to tensors
        H_his_tensor = torch.from_numpy(H_his_real).float()
        H_future_tensor = torch.from_numpy(H_future_real).float()
        
        return {
            'history': H_his_tensor,
            'future': H_future_tensor,
            'velocity': self.get_velocity_km_h(v_idx),
            'snr': snr
        }
    
    def get_velocity_km_h(self, v_idx):
        """Convert velocity index to km/h value."""
        return velocity_idx_to_km_h(v_idx)
    
    def get_active_velocities_km_h(self):
        """Get list of active velocities in km/h."""
        return [self.get_velocity_km_h(idx) for idx in self.active_velocity_indices]
    

# ===== Evaluation Utilities =====

def get_default_velocities():
    """Return the default velocity values list (0-60 km/h, step=5)."""
    return DEFAULT_VELOCITY_VALUES

def get_default_snr_values():
    """Return the default SNR values list (0-20 dB, step=1)."""
    return DEFAULT_SNR_VALUES

def evaluate_model_comprehensive(
    model, 
    dataset, 
    velocities: Optional[List[int]] = None, 
    snr_values: Optional[List[float]] = None,
    batch_size: int = 32
) -> Dict:
    """
    Evaluate model performance across different velocities and SNRs.
    
    Computes NMSE and SE metrics for all combinations of velocities and SNRs,
    preserving the prediction timestep dimension. Results can be extracted 
    using helper functions for analysis.
    
    Note: All velocity and SNR parameters use physical values (km/h and dB),
    not indices, for intuitive usage.
    
    Args:
        model: PyTorch model to evaluate
        dataset: FlexibleCSIDataset instance
        velocities: List of velocities (km/h) to test, defaults to 0-60km/h
        snr_values: List of SNR values (dB) to test, defaults to 0-20dB
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with results structured as:
        {
            'nmse': {
                'per_step': {(v, snr): [values_for_each_step]},  # v is velocity in km/h, snr is SNR in dB
                'avg': {(v, snr): average_value}
            },
            'se': {
                'est': {
                    'per_step': {(v, snr): [values_for_each_step]},
                    'avg': {(v, snr): average_value}
                },
                'true': {
                    'per_step': {(v, snr): [values_for_each_step]},
                    'avg': {(v, snr): average_value}
                }
            }
        }
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Use default values if not specified
    if velocities is None:
        velocities = DEFAULT_VELOCITY_VALUES
    if snr_values is None:
        snr_values = DEFAULT_SNR_VALUES
    
    # Initialize results structure
    results = {
        'nmse': {'per_step': {}, 'avg': {}},
        'se': {
            'est': {'per_step': {}, 'avg': {}},
            'true': {'per_step': {}, 'avg': {}}
        }
    }
    
    # Store dataset metadata
    results['metadata'] = {
        'snr_step': 1,  # dB per step
        'velocity_step': 5,  # km/h per step
        'velocity_range': [0, 60],  # km/h
        'snr_range': [0, 20],  # dB
        'velocities_used': velocities,
        'snr_values_used': snr_values,
        'Nr': dataset.Nr,
        'Nt': dataset.Nt
    }
    
    # Preserve original dataset state
    original_active_velocities = dataset.active_velocity_indices.copy()
    original_snr = dataset.current_snr
    
    # Test each velocity and SNR combination
    for v in velocities:
        v_idx = km_h_to_velocity_idx(v)  # Convert km/h to index
        dataset.set_active_velocities([v_idx])
        
        for snr_db in snr_values:
            # Set dataset to use this SNR
            dataset.set_snr(snr_db)
            
            # Create dataloader
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Collect predictions and targets
            all_preds = []
            all_targets = []
            
            print(f"Evaluating velocity {v} km/h, SNR {snr_db} dB...")
            
            with torch.no_grad():
                for batch in test_loader:
                    # Get data from batch dictionary
                    hist = batch['history']
                    future = batch['future']
                    
                    # Move data to device
                    if device.type != 'cpu':
                        hist = hist.to(device)
                        future = future.to(device)
                    
                    # Generate predictions
                    preds = model(hist)
                    
                    # Store results
                    all_preds.append(preds.cpu())
                    all_targets.append(future.cpu())
            
            # Combine all batches
            predictions = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Calculate metrics with physical values (km/h, dB) as keys
            key = (v, snr_db)
            
            # Calculate NMSE
            nmse_steps = calculate_nmse_per_step(predictions, targets)
            results['nmse']['per_step'][key] = nmse_steps.tolist()
            results['nmse']['avg'][key] = torch.mean(nmse_steps).item()
            
            # Calculate SE
            se_est_steps, se_true_steps = calculate_spectral_efficiency_per_step(
                h_est=predictions,
                h_true=targets,
                Nr=dataset.Nr,
                Nt=dataset.Nt,
                snr_db=snr_db,
                device='cpu'
            )
            
            # Store SE results
            results['se']['est']['per_step'][key] = se_est_steps.tolist()
            results['se']['est']['avg'][key] = torch.mean(se_est_steps).item()
            
            results['se']['true']['per_step'][key] = se_true_steps.tolist()
            results['se']['true']['avg'][key] = torch.mean(se_true_steps).item()
    
    # Restore original dataset state
    dataset.set_active_velocities(original_active_velocities)
    dataset.set_snr(original_snr if original_snr is not None else snr_values[0])
    
    print("Comprehensive evaluation complete.")
    return results

def extract_vs_velocity(results, metric='nmse', snr=10.0, step=None):
    """
    Extract results for analysis of performance vs velocity at fixed SNR.
    
    Args:
        results: Results dictionary from evaluate_model_comprehensive
        metric: 'nmse' or 'se'
        snr: SNR value to fix for analysis
        step: Specific prediction step to analyze (None = average across steps)
        
    Returns:
        Dictionary mapping velocities to metric values
    """
    extracted = {}
    if metric == 'nmse':
        data_source = results['nmse']['per_step' if step is not None else 'avg']
        for (v, s), value in data_source.items():
            if s == snr:
                extracted[v] = value[step] if step is not None else value
    elif metric == 'se':
        data_source = results['se']['est']['per_step' if step is not None else 'avg']
        for (v, s), value in data_source.items():
            if s == snr:
                extracted[v] = value[step] if step is not None else value
    
    return extracted

def extract_vs_snr(results, metric='nmse', velocity=30, step=None):
    """
    Extract results for analysis of performance vs SNR at fixed velocity.
    
    Args:
        results: Results dictionary from evaluate_model_comprehensive
        metric: 'nmse' or 'se'
        velocity: Velocity value to fix for analysis
        step: Specific prediction step to analyze (None = average across steps)
        
    Returns:
        Dictionary mapping SNR values to metric values
    """
    extracted = {}
    if metric == 'nmse':
        data_source = results['nmse']['per_step' if step is not None else 'avg']
        for (v, s), value in data_source.items():
            if v == velocity:
                extracted[s] = value[step] if step is not None else value
    elif metric == 'se':
        data_source = results['se']['est']['per_step' if step is not None else 'avg']
        for (v, s), value in data_source.items():
            if v == velocity:
                extracted[s] = value[step] if step is not None else value
    
    return extracted

def extract_vs_pred_step(results, metric='nmse', velocity=30, snr=10.0):
    """
    Extract results for analysis of performance vs prediction step at fixed velocity and SNR.
    
    Args:
        results: Results dictionary from evaluate_model_comprehensive
        metric: 'nmse' or 'se'
        velocity: Velocity value to fix for analysis
        snr: SNR value to fix for analysis
        
    Returns:
        List of metric values for each prediction step
    """
    key = (velocity, snr)
    if metric == 'nmse':
        if key in results['nmse']['per_step']:
            return results['nmse']['per_step'][key]
    elif metric == 'se':
        if key in results['se']['est']['per_step']:
            return results['se']['est']['per_step'][key]
    
    return None






if __name__ == "__main__":
    # Example usage
    file_path = "raw_data/UMa_Train.mat"
    
    # Create dataset
    dataset = FlexibleCSIDataset(
        file_path=file_path,
        seq_len=24,
        pred_len=6,
        is_training=True,
        snr_db=[0, 20],  # Random SNR between 0-20 dB in training
        active_velocity_indices=[0, 2, 4, 6, 8],  # 0, 10, 20, 30, 40 km/h
        verbose=True
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shapes - History: {sample['history'].shape}, Future: {sample['future'].shape}")
    print(f"Sample velocity: {sample['velocity']} km/h, SNR: {sample['snr']:.2f} dB")
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Example: Using set_active_velocities_km_h with actual velocity values
    print("\nSwitching to 0, 30, 60 km/h velocities:")
    dataset.set_active_velocities_km_h([0, 30, 60])  # More intuitive than indices
    
    # Example: Old way using indices (still works but less intuitive)
    print("\nSwitching to 50 km/h only using indices:")
    dataset.set_active_velocities([10])  # 50 km/h (index 10)
    
    # Example: Testing with specific SNR values
    print("\nChanging SNR to 15 dB:")
    dataset.set_snr(15.0)
    
    # Example: Creating dummy model and data for evaluation examples
    print("\nExample evaluation functions:")
    
    # Define a simple dummy model for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.linear = torch.nn.Linear(feature_dim, feature_dim)
            
        def forward(self, x):
            # For demonstration, predict future as same as last history step
            # In real usage, your actual model would be used
            last_step = x[:, -1:, :].repeat(1, 6, 1)  # Repeat last step for pred_len=6
            return last_step
    
    # Create dummy model
    feature_dim = dataset.Nr * dataset.Nt * 2
    dummy_model = DummyModel(feature_dim)
    
    print("\nExample of comprehensive evaluation:")
    print("This would normally be run on all velocities and SNRs,")
    print("but for demonstration we'll use a small subset.")
    
    # Demonstration with limited values
    test_velocities = [0, 30]  # km/h
    test_snrs = [0, 10]  # dB
    
    # In actual usage, you would do:
    # results = evaluate_model_comprehensive(
    #     model=your_model,
    #     dataset=test_dataset,
    #     velocities=list(range(0, 65, 5)),  # 0, 5, 10, ..., 60 km/h
    #     snr_values=list(range(0, 21, 2)),  # 0, 2, 4, ..., 20 dB
    #     batch_size=64
    # )
    
    # For demonstration only - using minimal subset
    results = evaluate_model_comprehensive(
        model=dummy_model,
        dataset=dataset,
        velocities=test_velocities,
        snr_values=test_snrs,
        batch_size=16
    )
    
    # Example: Extract NMSE vs velocity at fixed SNR
    print("\nExample: Extract NMSE vs velocity at SNR=10dB")
    nmse_vs_v = extract_vs_velocity(results, metric='nmse', snr=10.0)
    print(f"NMSE vs Velocity: {nmse_vs_v}")
    
    # Example: Extract SE vs SNR at fixed velocity
    print("\nExample: Extract SE vs SNR at velocity=30km/h")
    se_vs_snr = extract_vs_snr(results, metric='se', velocity=30)
    print(f"SE vs SNR: {se_vs_snr}")
    
    # Example: Extract NMSE vs prediction timestep
    print("\nExample: Extract NMSE vs prediction timestep at velocity=30km/h, SNR=10dB")
    nmse_vs_step = extract_vs_pred_step(results, metric='nmse', velocity=30, snr=10.0)
    print(f"NMSE per timestep: {nmse_vs_step}")
    
    print("\nNote: The comprehensive evaluation approach is more efficient")
    print("as it calculates all metrics at once and allows for flexible analysis.") 