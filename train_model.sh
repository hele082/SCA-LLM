#!/bin/bash

# ------------------------------------------------
# Training script for CSI prediction models
# ------------------------------------------------
#compared_models/LLM4CP/models/GPT4CP.py
# Default parameters
MODEL_CLASS="compared_models.LLM4CP.models.GPT4CP.GPT4CP"
MODEL_PARAMS=""

SEQ_LEN=24
PRED_LEN=6
BATCH_SIZE=1024
NUM_EPOCHS=500
LEARNING_RATE=1e-4
TRAIN_FILE="channel_data/2.4GHz/UMa_NLOS/2.4GHz_UMa_Train.mat"
SAVE_DIR="temp/checkpoints/"
DEVICE="cuda:2"
SEED=42
NUM_WORKERS=8  # Default to 0 for best H5py compatibility

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*)
      MODEL_CLASS="${1#*=}"
      ;;
    --seq_len=*)
      SEQ_LEN="${1#*=}"
      ;;
    --pred_len=*)
      PRED_LEN="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      ;;
    --epochs=*)
      NUM_EPOCHS="${1#*=}"
      ;;
    --lr=*)
      LEARNING_RATE="${1#*=}"
      ;;
    --data=*)
      TRAIN_FILE="${1#*=}"
      ;;
    --save_dir=*)
      SAVE_DIR="${1#*=}"
      ;;
    --device=*)
      DEVICE="${1#*=}"
      ;;
    --seed=*)
      SEED="${1#*=}"
      ;;
    --num_workers=*)
      NUM_WORKERS="${1#*=}"
      ;;
    --model_params=*)
      MODEL_PARAMS="${1#*=}"
      ;;
    --help)
      echo "Training script for CSI prediction models"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model=CLASS       Model class path (e.g., 'proposed_models.spa_gpt2.SPAGPT2')"
      echo "  --seq_len=N         Input sequence length"
      echo "  --pred_len=N        Prediction sequence length"
      echo "  --batch_size=N      Batch size"
      echo "  --epochs=N          Number of epochs"
      echo "  --lr=N              Learning rate"
      echo "  --data=FILE         Training data file path"
      echo "  --save_dir=DIR      Directory to save models"
      echo "  --device=DEVICE     Device to use (cuda:0, cpu)"
      echo "  --seed=N            Random seed"
      echo "  --num_workers=N     Number of DataLoader worker processes (default: 0)"
      echo ""
      echo "  --model_params=STR  Additional model parameters as key=value pairs separated by commas"
      echo "                      Example: --model_params=\"embed_dim=64,num_layers=4,dropout=0.1\""
      echo ""
      echo "Examples:"
      echo "  $0 --model=\"proposed_models.spa_gpt2.SPAGPT2\" --model_params=\"embed_dim=64,num_layers=4,num_heads=8\""
      echo "  $0 --model=\"compared_models.lstm.LSTM\" --model_params=\"hidden_size=128,num_layers=2\""
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
  shift
done

# Ensure save directory exists
mkdir -p "$SAVE_DIR"

# Print training configuration
echo "==========================================="
echo "          Training Configuration           "
echo "==========================================="
echo "Model class: $MODEL_CLASS"
echo "Sequence length: $SEQ_LEN"
echo "Prediction length: $PRED_LEN"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Training data: $TRAIN_FILE"
echo "Save directory: $SAVE_DIR"
echo "Device: $DEVICE"
echo "Random seed: $SEED"
echo "Num workers: $NUM_WORKERS"
if [ -n "$MODEL_PARAMS" ]; then
  echo "Model parameters: $MODEL_PARAMS"
fi
echo "==========================================="

# Build command with dynamic model parameters
TRAIN_CMD="python model_training.py \
  --model_class=\"$MODEL_CLASS\" \
  --seq_len=\"$SEQ_LEN\" \
  --pred_len=\"$PRED_LEN\" \
  --batch_size=\"$BATCH_SIZE\" \
  --num_epochs=\"$NUM_EPOCHS\" \
  --lr=\"$LEARNING_RATE\" \
  --train_file=\"$TRAIN_FILE\" \
  --save_dir=\"$SAVE_DIR\" \
  --device=\"$DEVICE\" \
  --seed=\"$SEED\" \
  --num_workers=\"$NUM_WORKERS\""

if [ -n "$MODEL_PARAMS" ]; then
  TRAIN_CMD="$TRAIN_CMD --model_params=\"$MODEL_PARAMS\""
fi

# Print and execute command
echo "Executing training with command:"
echo "$TRAIN_CMD"
eval "$TRAIN_CMD"

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  # Find the latest model file
  LATEST_MODEL=$(ls -t "$SAVE_DIR"/*.pth 2>/dev/null | head -n 1)
  if [ -n "$LATEST_MODEL" ]; then
    echo "Model saved to: $LATEST_MODEL"
    echo ""
    echo "You can evaluate the model using:"
    echo "python model_evaluation.py --model_path=\"$LATEST_MODEL\" --model_class=\"$MODEL_CLASS\" --data_file_path=\"channel_data/UMa_Test.mat\""
  else
    echo "Warning: Could not find saved model file"
  fi
else
  echo "Training process failed, check logs for errors"
fi 