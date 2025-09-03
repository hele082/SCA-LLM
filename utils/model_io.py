import torch
import os
import inspect
import functools
from typing import Callable, Type



def print_model_info(model: torch.nn.Module):
    """
    Prints the total number of parameters and the number of learnable parameters in the model.

    Args:
        model (torch.nn.Module): The model to inspect.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1e6:.3f}M")
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {learnable_params / 1e6:.3f}M")
    print(f"Model settings: {model._init_args}")


def save_model(model: torch.nn.Module, full_save_path: str, current_best_val_loss: float):
    """
    Saves the model state_dict, model __init__ args (expected to be stored in `model._init_args`),
    and the current best validation loss to a specified file path.
    Requires the model instance to have an `_init_args` attribute containing the arguments
    passed to its `__init__` method (e.g., set by calling `capture_init_args(self, locals())`
    at the end of `__init__`).
    Overwrites the existing file if it exists.

    Args:
        model (torch.nn.Module): The model to save, which should have the `_init_args` attribute set.
        full_save_path (str): The exact path (including filename) to save the checkpoint.
        current_best_val_loss (float): The validation loss corresponding to this model state.

    Raises:
        ValueError: If the model instance does not have a valid `_init_args` attribute.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    model_state_dict = model.state_dict()

    # Check for the _init_args attribute set by the helper function (or potentially decorator)
    if hasattr(model, '_init_args') and isinstance(model._init_args, dict):
        model_settings = model._init_args
    else:
        # If not found or invalid, raise an error
        error_msg = (
            f"Error saving model {model.__class__.__name__}: Missing or invalid `_init_args` attribute. "
            f"Please ensure the model's `_init_args` attribute was correctly set (e.g., by calling `capture_init_args` in `__init__`)."
        )
        print(error_msg) # Print the error before raising
        raise ValueError(error_msg)

    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model_state_dict,
        'model_settings': model_settings, # Settings captured by the decorator
        'best_val_loss': current_best_val_loss, # The validation loss achieved by this state
    }

    # Save the checkpoint
    try:
        torch.save(checkpoint, full_save_path)
        print(f"Model checkpoint saved successfully to {full_save_path} (Val Loss: {current_best_val_loss:.6f})")
    except Exception as e:
        # Print any saving errors
        print(f"Error during torch.save to {full_save_path}: {e}")
        raise # Re-raise the exception after printing


def load_model(model_class: Type[torch.nn.Module],
               path: str,
               map_location: str = 'cpu') -> torch.nn.Module:
    """
    Loads model settings and state dict from a checkpoint file (.pth),
    instantiates the model, and loads the weights.
    Optionally attaches 'best_val_loss_loaded' to the model if present in the checkpoint.

    Args:
        model_class (Type[torch.nn.Module]): The class of the model to load (e.g., `MyModel`).
        path (str): Path to the checkpoint file (.pth).
        map_location (str): The device to load the model onto ('cpu', 'cuda:0', etc.). Defaults to 'cpu'.

    Returns:
        torch.nn.Module: The instantiated model with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        KeyError: If required keys ('model_settings', 'model_state_dict') are missing.
        TypeError: If 'model_settings' is not a dict or instantiation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False) # Load directly to target device if needed later
    except Exception as e:
        raise IOError(f"Failed to load checkpoint file {path}: {e}")

    # Validate checkpoint structure
    if 'model_settings' not in checkpoint:
         raise KeyError(f"Checkpoint file {path} is missing 'model_settings'")
    if not isinstance(checkpoint['model_settings'], dict):
        raise TypeError(f"Expected 'model_settings' in {path} to be a dict, got {type(checkpoint['model_settings'])}")
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint file {path} is missing 'model_state_dict'")

    model_settings = checkpoint['model_settings']

    # Filter settings to match model's __init__ signature
    try:
        init_signature = inspect.signature(model_class.__init__)
        init_params = {param.name for param in init_signature.parameters.values()
                       if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]}
        # Include 'self' if needed, though typically handled by Python
        init_params.discard('self')
        init_kwargs = {key: value for key, value in model_settings.items() if key in init_params}
    except Exception as e:
        raise RuntimeError(f"Failed to inspect signature of {model_class.__name__}.__init__: {e}")

    # Instantiate the model
    try:
        model = model_class(**init_kwargs)
        model.model_settings = model_settings # Store the loaded settings on the model instance
    except Exception as e:
        raise TypeError(f"Failed to instantiate model {model_class.__name__} with filtered args {init_kwargs} "
                        f"(from settings {model_settings}): {e}")

    # Load the state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict into {model_class.__name__}: {e}")

    # Store best_val_loss if available
    if 'best_val_loss' in checkpoint:
        model.best_val_loss_loaded = checkpoint['best_val_loss'] # Store as an attribute
        print(f"  (Loaded best_val_loss from checkpoint: {model.best_val_loss_loaded:.6f})")

    model.to(map_location) # Ensure model is on the correct device

    model_name = model_settings.get('name', model_class.__name__) # Use name from settings if available
    print(f"Model {model_name} ({model_class.__name__}) instantiated with settings and loaded from {path} to {map_location}")

    return model


# --- Helper function to capture __init__ args without a decorator --- START ---
def capture_init_args(instance, init_locals):
    """
    Captures arguments passed to the __init__ method by inspecting the signature
    and matching parameter names found in the provided locals() dictionary.
    Stores the captured arguments in instance._init_args.

    This function should be called at the very end of the __init__ method:
    capture_init_args(self, locals())

    Args:
        instance: The instance of the class being initialized (self).
        init_locals (dict): The dictionary returned by locals() just before calling
                          this function within the __init__ method.
    """
    init_func = instance.__class__.__init__
    if init_func is object.__init__:
        # Avoid trying to inspect the basic object.__init__
        instance._init_args = {}
        return

    try:
        sig = inspect.signature(init_func)
    except ValueError: # Happens for some built-in types or C extensions
        instance._init_args = {} # Cannot inspect signature
        return

    args_to_store = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        # Check if the parameter name exists in the locals dictionary
        if name in init_locals:
            args_to_store[name] = init_locals[name]
        # If not in locals, it might be using its default value
        elif param.default is not inspect.Parameter.empty:
            args_to_store[name] = param.default
        # else: parameter was not provided and has no default (should ideally not happen if init called correctly)
            # print(f"Warning: Parameter '{name}' not found in locals and has no default for {instance.__class__.__name__}.__init__")

    instance._init_args = args_to_store
# --- Helper function to capture __init__ args without a decorator --- END ---

