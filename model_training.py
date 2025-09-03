import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from datetime import datetime
import inspect

from csi_dataset import FlexibleCSIDataset, get_default_velocities, get_default_snr_values
from utils.model_io import print_model_info
from utils.training_utils import train_model
from utils.evaluation_metrics import NMSELoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train CSI prediction models")
    parser.add_argument("--model_class", type=str, required=True, 
                        help="Model class import path (e.g., 'proposed_models.spa_gpt2.SPAGPT2')")
    parser.add_argument("--model_params", type=str, default="",
                        help="Additional model parameters as key=value pairs separated by commas")
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length for history")
    parser.add_argument("--pred_len", type=int, default=6, help="Prediction length for future")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (e.g., 'cuda:0', 'cpu'). If None, auto-select.")
    parser.add_argument("--train_file", type=str, default="channel_data/UMa_Train.mat",
                        help="Path to training data file")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of training data to total data")
    parser.add_argument("--save_dir", type=str, default="temp/checkpoints/",
                        help="Directory to save model")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum improvement for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading (0 = single process)")
    
    return parser.parse_args()

def parse_model_params(param_str):
    """
    Parse model parameters from a string of key=value pairs separated by commas.
    
    Args:
        param_str (str): String containing key=value pairs separated by commas
        
    Returns:
        dict: Dictionary of parameter names and values with appropriate types
    """
    if not param_str:
        return {}
    
    params = {}
    for pair in param_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert value to appropriate type
            try:
                # Try boolean
                if value.lower() in ["true", "false"]:
                    params[key] = value.lower() == "true"
                # Try number
                elif "." in value or "e" in value.lower():
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # Keep as string if conversion fails
                params[key] = value
    
    return params

def load_model_class(model_path):
    """
    Dynamically load a model class from a string path.
    
    Args:
        model_path (str): Path to model class (e.g., 'proposed_models.spa_gpt2.SPAGPT2')
        
    Returns:
        model_class: The loaded model class
    """
    try:
        module_path, class_name = model_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import model class {model_path}: {e}")

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load dataset
    print("--- Loading Dataset ---")
    dataset = FlexibleCSIDataset(
        file_path=args.train_file,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        is_training=True,
        snr_db=[0, 20],  # Random SNR for training in range [0,20] dB
        verbose=True
    )
    
    # Get feature size
    feature_size = dataset[0]['history'].shape[1]
    print(f"Feature size: {feature_size}")
    
    # Split dataset
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Splitting dataset (size {len(dataset)}) into train ({train_size}) and validation ({val_size})")
    
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid train/validation split sizes ({train_size}/{val_size}).")
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != 'cpu'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type != 'cpu'
    )
    
    # Load model class and instantiate model
    print("--- Loading Model ---")
    model_class = load_model_class(args.model_class)
    
    # Get model class name for saving
    model_name = args.model_class.split('.')[-1]
    
    # Parse additional model parameters from command line
    model_params = parse_model_params(args.model_params)
    if model_params:
        print(f"Model parameters: {model_params}")
    
    # Simply pass the user-specified parameters to the model
    # Let the model itself handle defaults and required parameters
    print(f"Initializing model with parameters: {model_params}")
    model = model_class(**model_params).to(device)
    
    print_model_info(model)
    
    # Setup training components
    print("--- Setting up Training Components ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = NMSELoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Setup save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_sl{args.seq_len}_pl{args.pred_len}_{timestamp}"
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup save paths
    full_save_path = os.path.join(args.save_dir, f"{run_name}.pth")
    
    # Setup TensorBoard
    tb_log_dir = os.path.join('runs', run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    print(f"Model will be saved to: {full_save_path}")
    print(f"TensorBoard logs: {tb_log_dir}")
    
    # Train model
    print("--- Starting Training ---")
    training_result = train_model(
        model=model,
        train_loader=train_loader,
        vali_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        full_save_path=full_save_path,
        num_epochs=args.num_epochs,
        device=device,
        initial_best_val_loss=float('inf'),
        writer=writer,
        early_stopping_patience=args.early_stop,
        min_delta=args.min_delta
    )
    
    # Training complete
    print("--- Training Complete ---")
    print(f"Best validation loss: {training_result['best_val_loss']:.6f}")
    print(f"Training completed in {training_result['epochs_completed']} epochs")
    print(f"Model saved to: {full_save_path}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main() 