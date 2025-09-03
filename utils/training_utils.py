import torch
import numpy as np
from tqdm import tqdm
import json
import os
from typing import Callable, Optional, Dict, Any, Type, Union, List
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.model_io import save_model, print_model_info


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: str
) -> float:
    """
    Performs a single training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function.
        device (str): The device to perform training on ('cpu', 'cuda:0', etc.).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set model to training mode
    epoch_losses = []
    epoch_weights = []   # Use epoch weight (e.g., number of samples) for accurate averaging
    pbar = tqdm(train_loader, desc="Training", leave=False) # leave=False for cleaner multi-epoch output

    for batch in pbar:
        # Get history and future data from the batch dictionary
        observation = batch['history']
        target = batch['future']
        batch_size = observation.shape[0]
        
        # Move data to the target device
        try:
            observation = observation.to(device)
            target = target.to(device)
        except Exception as e:
            print(f"\nError during data moving in training: {e}")
            print(f"Observation type: {type(observation)}, Target type: {type(target)}")
            raise e

        # Standard training steps
        optimizer.zero_grad()
        try:
            prediction = model(observation)
            loss = criterion(prediction, target)
        except Exception as e:
             print(f"\nError during model forward pass or loss calculation in training: {e}")
             print(f"Observation shape: {observation.shape}, Target shape: {target.shape}")
             raise e

        # Check for invalid loss values before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN or Inf loss detected during training (value: {loss.item()}). Skipping backward pass for this batch.")
            continue # Skip batch update

        # Backward pass and optimizer step
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"\nError during backward pass or optimizer step: {e}")
            raise e

        # Calculate weighted loss and check for finiteness *before* appending
        weighted_loss = loss.item() * batch_size
        if np.isfinite(weighted_loss):
            epoch_losses.append(weighted_loss)
            epoch_weights.append(batch_size)
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'}) # Keep showing original batch loss
        else:
            print(f"\nWarning: Finite loss {loss.item():.4f} resulted in non-finite weighted loss ({weighted_loss}). Skipping batch contribution to epoch loss.")

    # Calculate average loss for the epoch using directly accumulated valid weights
    total_weight = np.sum(epoch_weights)
    if total_weight == 0:
        print("\nWarning: No valid batches with finite weighted loss processed in training epoch.")
        return float('nan') # Return NaN if no weights were accumulated

    # Sum of weighted losses / sum of weights
    epoch_loss = np.sum(epoch_losses) / total_weight
    return epoch_loss


def validate(
    model: torch.nn.Module,
    vali_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str
) -> float:
    """
    Performs validation on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        vali_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        device (str): The device to perform validation on.

    Returns:
        float: The average validation loss. Returns float('inf') if validation fails or no valid batches.
    """
    model.eval() # Set model to evaluation mode
    epoch_losses = []
    epoch_weights = []

    pbar = tqdm(vali_loader, desc="Validating", leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for batch in pbar:
            # Get history and future data from the batch dictionary
            observation = batch['history']
            target = batch['future']
            batch_size = observation.shape[0]

            # Move data
            try:
                observation = observation.to(device)
                target = target.to(device)
            except Exception as e:
                print(f"\nError during data moving in validation: {e}")
                raise e

            # Forward pass and loss calculation
            try:
                prediction = model(observation)
                loss = criterion(prediction, target)
            except Exception as e:
                print(f"\nError during model forward pass or loss calculation in validation: {e}")
                print(f"Observation shape: {observation.shape}, Target shape: {target.shape}")
                raise e

            # Check for invalid loss values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected during validation (value: {loss.item()}). Skipping batch.")
                continue # Skip batch

            # Calculate weighted loss and check finiteness before accumulating
            weighted_loss = loss.item() * batch_size
            if np.isfinite(weighted_loss):
                epoch_losses.append(weighted_loss)
                epoch_weights.append(batch_size)
                pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
            else:
                print(f"\nWarning: Finite validation loss {loss.item():.4f} resulted in non-finite weighted loss ({weighted_loss}). Skipping batch contribution.")

    # Calculate average loss from accumulated lists
    total_weight = np.sum(epoch_weights)
    if total_weight == 0:
        print("\nWarning: No valid batches with finite weighted loss processed during validation.")
        return float('nan')

    total_loss = np.sum(epoch_losses)
    average_loss = total_loss / total_weight
    return average_loss


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    vali_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    scheduler: Optional[_LRScheduler], 
    full_save_path: str,
    num_epochs: int,
    device: str,
    initial_best_val_loss: float = float('inf'),
    writer: Optional[SummaryWriter] = None,
    early_stopping_patience: Optional[int] = None,
    min_delta: float = 0.0,
    save_history: bool = True,
    history_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a model over multiple epochs with validation, checkpointing the best model,
    learning rate scheduling, optional TensorBoard logging, and optional early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        vali_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler (or None).
        full_save_path (str): The *exact* file path to save the best model checkpoint.
                              This file will be overwritten when a better model is found.
        num_epochs (int): Number of epochs to train for.
        device (str): Device for training ('cpu', 'cuda:0', etc.).
        initial_best_val_loss (float): The best validation loss known before starting this training run.
                                      Defaults to infinity. Used to decide if the first epoch's result
                                      is already an improvement worth saving.
        writer (Optional[SummaryWriter]): TensorBoard SummaryWriter instance (or None).
        early_stopping_patience (Optional[int]): Patience for early stopping (epochs). None or 0 disables it.
        min_delta (float): Minimum change in val_loss to qualify as improvement for early stopping.
        save_history (bool): Whether to save training history to a JSON file.
        history_path (Optional[str]): Path to save training history JSON (if save_history is True).
                                    If None, will be derived from full_save_path.

    Returns:
        Dict[str, Any]: Dictionary containing training history and best validation loss.
    """
    # Setup
    best_val_loss = initial_best_val_loss
    patience_counter = 0
    if early_stopping_patience and early_stopping_patience <= 0:
        early_stopping_patience = None  # Disable if <= 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    print(f"Starting training for {num_epochs} epochs")
    print(f"Using {device} device")
    print(f"Early stopping: {'Enabled' if early_stopping_patience else 'Disabled'}")
    if early_stopping_patience:
        print(f"  Patience: {early_stopping_patience}, Min delta: {min_delta}")

    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Training phase
        try:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            if not np.isfinite(train_loss):
                print(f"Warning: Non-finite train loss ({train_loss}) in epoch {epoch+1}. "
                      f"Continuing training but this epoch's training results may be unreliable.")
                train_loss = float('nan') # Use NaN for history to indicate problematic epoch
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {e}")
            train_loss = float('nan')
            # We continue to validation to see if we should stop training

        # Validation phase
        try:
            val_loss = validate(model, vali_loader, criterion, device)
            if not np.isfinite(val_loss):
                print(f"Warning: Non-finite validation loss ({val_loss}) in epoch {epoch+1}. "
                      f"Continuing training but validation results for this epoch are unreliable.")
                val_loss = float('nan')
        except Exception as e:
            print(f"Error during validation in epoch {epoch+1}: {e}")
            val_loss = float('nan')
            # If validation fails, we typically can't make a save decision

        # Record history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))

        # Update TensorBoard if available
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Learning rate', current_lr, epoch)

        # Calculate delta (improvement) before potentially updating best_val_loss
        delta = best_val_loss - val_loss  # Positive if improved, negative if worse
        
        # Check for improvement
        is_best = False
        if np.isfinite(val_loss) and (np.isnan(best_val_loss) or val_loss < best_val_loss - min_delta):
            is_best = True
            # Save the previous best for proper delta calculation
            previous_best = best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model if validation loss improved
            try:
                save_model(model, full_save_path, best_val_loss)
            except Exception as e:
                print(f"Error during model saving in epoch {epoch+1}: {e}")
                # Continue training even if saving fails
        else:
            if early_stopping_patience:
                patience_counter += 1
            
        # Print epoch summary
        if delta > min_delta:  # Significant improvement
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f} "
                  f"(delta: {delta:.6f}, min_delta: {min_delta:.6f}, improved, best: {best_val_loss:.6f})")
        elif delta > 0:  # Improvement but not significant enough
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f} "
                  f"(delta: {delta:.6f}, min_delta: {min_delta:.6f}, better but not saved, best: {best_val_loss:.6f})")
        else:  # No improvement
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f} "
                  f"(delta: {delta:.6f}, min_delta: {min_delta:.6f}, no improvement, best: {best_val_loss:.6f})")
              
        # Step learning rate scheduler if provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Check early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {patience_counter} epochs)")
            break

    # Save training history if requested
    if save_history:
        if history_path is None:
            # Derive history path from model save path if not provided
            history_path = os.path.splitext(full_save_path)[0] + "_training_history.json"
        
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Training history saved to {history_path}")
        except Exception as e:
            print(f"Error saving training history: {e}")

    # Final summary
    print(f"Training completed in {len(history['train_loss'])} epochs")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print_model_info(model)
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'epochs_completed': len(history['train_loss'])
    } 