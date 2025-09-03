import argparse
import os
import torch
import h5py
import json
import pickle
import numpy as np
from datetime import datetime

from csi_dataset import FlexibleCSIDataset, evaluate_model_comprehensive, get_default_velocities, get_default_snr_values
from utils.model_io import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_class", type=str, required=True, help="Model class import path (e.g., 'proposed_models.spa_gpt2.SPAGPT2')")
    parser.add_argument("--data_file_path", type=str, default="channel_data/2.4GHz/UMa_NLOS/2.4GHz_UMa_Test.mat", help="Path to test data file")
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length for history")
    parser.add_argument("--pred_len", type=int, default=6, help="Prediction length for future")
    parser.add_argument("--test_velocities", type=str, default="0,5,10,15,20,25,30,35,40,45,50,55,60", 
                        help="Comma-separated list of velocities (km/h) to test")
    parser.add_argument("--test_snrs", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20", 
                        help="Comma-separated list of SNR values (dB) to test")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (e.g., 'cuda:0', 'cpu'). If None, auto-select.")
    parser.add_argument("--output_dir", type=str, default="output/results", help="Directory to save results")
    
    return parser.parse_args()

def save_results(results, args, model_path, output_dir):
    """Save results in both machine-friendly and human-readable formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename from model path 
    model_name = os.path.basename(model_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_name}_eval_{timestamp}"
    
    # 1. Save as MATLAB .mat file (machine-friendly)
    matlab_path = os.path.join(output_dir, f"{base_filename}.mat")
    save_matlab_format(results, args, matlab_path)
    
    # 2. Save as Python pickle (machine-friendly)
    pickle_path = os.path.join(output_dir, f"{base_filename}.pkl")
    save_pickle_format(results, args, pickle_path)
    
    # 3. Save as JSON (human-readable)
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    save_json_format(results, args, json_path)
    
    print(f"Results saved to:")
    print(f"  MATLAB: {matlab_path}")
    print(f"  Pickle: {pickle_path}")
    print(f"  JSON: {json_path}")
    
    return {
        'matlab': matlab_path,
        'pickle': pickle_path,
        'json': json_path
    }

def save_matlab_format(results, args, output_path):
    """Save results in MATLAB-compatible format"""
    import scipy.io
    
    # Create velocity and SNR arrays
    vel_values = sorted(list(set([v for (v, _) in results['nmse']['avg'].keys()])))
    snr_values = sorted(list(set([s for (_, s) in results['nmse']['avg'].keys()])))
    
    # Create matrices for easy MATLAB plotting
    nmse_matrix = np.full((len(vel_values), len(snr_values)), np.nan)
    se_est_matrix = np.full((len(vel_values), len(snr_values)), np.nan)
    se_true_matrix = np.full((len(vel_values), len(snr_values)), np.nan)
    
    # Fill in matrices
    for i, v in enumerate(vel_values):
        for j, s in enumerate(snr_values):
            if (v, s) in results['nmse']['avg']:
                nmse_matrix[i, j] = results['nmse']['avg'][(v, s)]
            
            if (v, s) in results['se']['est']['avg']:
                se_est_matrix[i, j] = results['se']['est']['avg'][(v, s)]
                
            if (v, s) in results['se']['true']['avg']:
                se_true_matrix[i, j] = results['se']['true']['avg'][(v, s)]
    
    # Store per-step metrics in a more accessible way
    nmse_step_dict = {}
    se_est_step_dict = {}
    se_true_step_dict = {}
    
    for (v, s), values in results['nmse']['per_step'].items():
        nmse_step_dict[f'v{v}_s{s}'] = np.array(values)
    
    for (v, s), values in results['se']['est']['per_step'].items():
        se_est_step_dict[f'v{v}_s{s}'] = np.array(values)
        
    for (v, s), values in results['se']['true']['per_step'].items():
        se_true_step_dict[f'v{v}_s{s}'] = np.array(values)
    
    # Save to MATLAB .mat file
    scipy.io.savemat(output_path, {
        'velocity_values': np.array(vel_values),
        'snr_values': np.array(snr_values),
        'nmse_matrix': nmse_matrix,
        'se_est_matrix': se_est_matrix,
        'se_true_matrix': se_true_matrix,
        'nmse_steps': nmse_step_dict,
        'se_est_steps': se_est_step_dict,
        'se_true_steps': se_true_step_dict,
        'config': {
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'model_class': args.model_class
        }
    })

def save_pickle_format(results, args, output_path):
    """Save results as Python pickle for easy reloading"""
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'args': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f)

def save_json_format(results, args, output_path):
    """Save human-readable results as JSON"""
    # Convert results to JSON-serializable format
    json_results = {
        'args': vars(args),
        'nmse': {
            'average': {},
            'per_step': {}
        },
        'se': {
            'estimated': {
                'average': {},
                'per_step': {}
            },
            'true': {
                'average': {},
                'per_step': {}
            }
        }
    }
    
    # Convert tuples to strings for JSON keys
    for (v, s), value in results['nmse']['avg'].items():
        json_results['nmse']['average'][f"{v}km/h_{s}dB"] = value
    
    for (v, s), values in results['nmse']['per_step'].items():
        json_results['nmse']['per_step'][f"{v}km/h_{s}dB"] = [float(x) for x in values]
    
    for (v, s), value in results['se']['est']['avg'].items():
        json_results['se']['estimated']['average'][f"{v}km/h_{s}dB"] = value
        
    for (v, s), values in results['se']['est']['per_step'].items():
        json_results['se']['estimated']['per_step'][f"{v}km/h_{s}dB"] = [float(x) for x in values]
        
    for (v, s), value in results['se']['true']['avg'].items():
        json_results['se']['true']['average'][f"{v}km/h_{s}dB"] = value
        
    for (v, s), values in results['se']['true']['per_step'].items():
        json_results['se']['true']['per_step'][f"{v}km/h_{s}dB"] = [float(x) for x in values]
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

def main():
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load model class dynamically
    import importlib
    module_path, class_name = args.model_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Load model
    model = load_model(model_class, args.model_path, map_location=device)
    model.eval()
    
    # Parse test velocities and SNRs
    test_velocities = [int(v) for v in args.test_velocities.split(',')]
    test_snrs = [float(s) for s in args.test_snrs.split(',')]
    
    # Prepare dataset
    dataset = FlexibleCSIDataset(
        file_path=args.data_file_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        is_training=False,
        snr_db=10.0,  # Default SNR, will be changed during evaluation
        verbose=True
    )
    
    # Run comprehensive evaluation
    print(f"Evaluating model with velocities: {test_velocities} km/h")
    print(f"SNR values: {test_snrs} dB")
    
    results = evaluate_model_comprehensive(
        model=model,
        dataset=dataset,
        velocities=test_velocities,
        snr_values=test_snrs,
        batch_size=args.batch_size
    )
    
    # Save results
    save_results(results, args, args.model_path, args.output_dir)

if __name__ == "__main__":
    main() 