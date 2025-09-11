# SCA-LLM: Spectral-Attentive Channel Prediction with Large Language Models in MIMO-OFDM

[![arXiv](https://img.shields.io/badge/arXiv-Your_Paper_ID-b31b1b.svg)](https://arxiv.org/abs/2509.08139)

This repository contains the implementation for the paper: [SCA-LLM: Spectral-Attentive Channel Prediction with Large Language Models in MIMO-OFDM](https://arxiv.org/abs/2509.08139).

We firmly believe that openness and sharing are the core driving forces for the advancement of the academic community. To promote research transparency and result reproducibility, we are open-sourcing the complete implementation code for this paper. Although as part of a research project, the code may have some imperfections, we sincerely hope it can provide a valuable reference and help for researchers in related fields.

## Note on Naming

Please note that the core model proposed in our paper is named **SCA-LLM**. However, during the initial development phase of this project, the model was temporarily named `SPAGPT2`. Therefore, you will find that some of the source code files, model class names, and saved checkpoint files (e.g., in `proposed_models/`, `ablation_experiment/ckpts/`) still use the `SPAGPT2` naming convention. This is a historical artifact and does not affect the implementation or the results presented in the paper.

Similarly, the baseline model from the [LLM4CP](https://github.com/liuboxun/LLM4CP) project, which we have included for comparison, uses the name `GPT4CP` within its source code.

## Project Structure

```
├── proposed_models/        # Custom model implementations for SCA-LLM
├── compared_models/        # Baseline models for comparison
├── utils/                  # Utility functions and helpers
├── channel_data/          # CSI dataset files
├── temp/                  # Temporary files and checkpoints
├── runs/                  # TensorBoard logs
├── Results/              # Evaluation results of various experiments
├── Figures/              # Generated visualization plots
├── model_training.py     # Training script
├── model_evaluation.py   # Evaluation script
├── result_visualization.py # Visualization tools
├── csi_dataset.py        # Dataset handling
├── train_model.sh        # Training workflow script
├── run_visualization_*.sh # Visualization scripts
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SCA-LLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The CSI dataset used for this project is too large to be hosted on GitHub. Please download it from one of the following links:

**[>> Google Drive <<](https://drive.google.com/file/d/1fZbpsHKDjFYntIostH2jFfX1wC_OUUdj/view?usp=sharing)**

**[>> Baidu Net Disk <<](https://pan.baidu.com/s/1_Ynj3kZ3IvvCkQOXTvwyWA?pwd=scal)**

After downloading, place the `.mat` files into the `channel_data/` directory. The expected structure is:
```
channel_data/2.4GHz/UMa_NLOS
├── 2.4GHz_UMa_Test.mat
└── 2.4GHz_UMa_Train.mat
```

The dataset files are in MATLAB's v7.3 format, which uses HDF5 internally to store large arrays. The main data is stored under the following key and shape:
- Key: 'channel_data'
- Shape: (Nv, Nu, T, Nsc, Nr, Nt) where:
  - Nv: Number of velocities (0-60km/h, step: 5km/h)
  - Nu: Number of users
  - T: Number of time slots
  - Nsc: Number of subcarriers
  - Nr: Number of receive antennas
  - Nt: Number of transmit antennas

## Model Checkpoints

The trained model checkpoints used to generate the results in the paper are provided in this repository.
- The main models (SCA-LLM, baselines) are located in `checkpoints/2.4GHz/`.
- Models for the ablation study are in `ablation_experiment/ckpts/`.

These models can be directly used for evaluation to reproduce the paper's findings.

## How to Reproduce Paper Results

This section provides a step-by-step guide to reproduce the main performance comparison figures for the UMa scenario presented in the paper.

### Step 1: Setup Environment and Dataset

1.  Follow the **Installation** instructions to set up the environment and install all dependencies.
2.  Follow the **Dataset Preparation** instructions to download the dataset and place it in the `channel_data/` directory.

### Step 2: Run Evaluation

Run the `model_evaluation.py` script for each of the pre-trained models. The results will be saved to the `Results/UMa/` directory.

Here are the example commands for the main models:

```bash
# Evaluate SCA-LLM (SPAGPT2)
python model_evaluation.py \
    --model_path="ablation_experiment/ckpts/SPAGPT2_sl24_pl6_20250407_092052.pth" \
    --model_class="proposed_models.spa_gpt2.SPAGPT2" \
    --data_file_path="channel_data/2.4GHz/UMa_NLOS/2.4GHz_UMa_Test.mat" \
    --output_dir="Results/UMa"

# Evaluate GRU
python model_evaluation.py \
    --model_path="checkpoints/2.4GHz/GRU_sl24_pl6_20250407_054721.pth" \
    --model_class="compared_models.gru.GRU" \
    --data_file_path="channel_data/2.4GHz/UMa_NLOS/2.4GHz_UMa_Test.mat" \
    --output_dir="Results/UMa"

# ... (run for other baseline models as needed)
```

### Step 3: Generate Visualization

After the evaluation for all desired models is complete, run the provided visualization script to generate the plots.

```bash
./run_visualization_UMa.sh
```

The final figures will be saved in the `Figures/UMa/` directory, which should match the results presented in the paper.

## Training Models

### Using the Training Script

The main training script `model_training.py` supports various parameters:

```bash
python model_training.py \
    --model_class="proposed_models.your_model.YourModel" \
    --model_params="param1=value1,param2=value2" \
    --seq_len=24 \
    --pred_len=6 \
    --batch_size=2048 \
    --num_epochs=500 \
    --lr=1e-4 \
    --train_file="channel_data/UMa_Train.mat" \
    --train_ratio=0.8 \
    --save_dir="temp/checkpoints/" \
    --early_stop=10 \
    --min_delta=0.001
```

Key parameters:
- `model_class`: Python path to your model class
- `model_params`: Additional model parameters as key=value pairs
- `seq_len`: Input sequence length (history)
- `pred_len`: Output prediction length (future)
- `batch_size`: Training batch size
- `num_epochs`: Maximum training epochs
- `lr`: Learning rate
- `train_file`: Path to training data
- `train_ratio`: Train/validation split ratio
- `save_dir`: Directory to save checkpoints
- `early_stop`: Early stopping patience
- `min_delta`: Minimum improvement for early stopping

### Using the Training Script

For convenience, use the `train_model.sh` script:

```bash
./train_model.sh \
    --model="proposed_models.your_model.YourModel" \
    --data_file="channel_data/UMa_Train.mat" \
    --seq_len=24 \
    --pred_len=6
```

## Model Evaluation

### Comprehensive Evaluation

The evaluation script `model_evaluation.py` performs comprehensive testing across different velocities and SNRs:

```bash
python model_evaluation.py \
    --model_path="temp/checkpoints/model.pth" \
    --model_class="proposed_models.your_model.YourModel" \
    --data_file_path="channel_data/UMa_Test.mat" \
    --seq_len=24 \
    --pred_len=6 \
    --test_velocities="0,5,10,15,20,25,30,35,40,45,50,55,60" \
    --test_snrs="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
    --batch_size=1024 \
    --output_dir="Results"
```

## Visualization

### Using the Visualization Script

The `result_visualization.py` script supports various plot types:

```bash
python result_visualization.py \
    --results="Results/model_eval_*.pkl" \
    --labels="Model1,Model2" \
    --plot_type="all" \
    --metric="both" \
    --velocity=30 \
    --snr=10.0 \
    --output_dir="Figures" \
    --se_type="both" \
    --save_format="png"
```

Plot types:
- `velocity`: Performance vs. velocity at fixed SNR
- `snr`: Performance vs. SNR at fixed velocity
- `step`: Performance vs. prediction step
- `all`: Generate all plot types

### Using Visualization Scripts

For convenience, use the provided visualization scripts:

```bash
# For UMa scenario
./run_visualization_UMa.sh

# For UMi scenario with zero-shot evaluation
./run_visualization_zero_shot_UMi.sh
```

### Visualization Output

The visualization scripts generate:
1. High-quality plots in specified format (PNG, PDF, etc.)
2. MATLAB-compatible data files for further customization
3. MATLAB scripts for recreating and customizing plots

## Creating Custom Models

To create a new model:

1. Create a new file in `proposed_models/` (e.g., `your_model.py`)
2. Define your model class following this template:

```python
from utils.model_io import capture_init_args

class YourModel(torch.nn.Module):
    def __init__(self, seq_len, pred_len, feature_size, **kwargs):
        super(YourModel, self).__init__()
        capture_init_args(self, locals())
        
        # Your model implementation
        self.encoder = ...
        self.decoder = ...
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_size)
        # output shape: (batch_size, pred_len, feature_size)
        return predicted_output
```

Key requirements:
- Must accept `seq_len`, `pred_len`, and `feature_size` parameters
- Call `capture_init_args(self, locals())` after `super().__init__()`
- Input shape: (batch_size, seq_len, feature_size)
- Output shape: (batch_size, pred_len, feature_size)

## Citation

Our paper is currently under review. We will update this section with the complete citation information upon acceptance.
If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{SCALLM,
  author    = {Ke He and
               Le He and
               Lisheng Fan and
               Xianfu Lei and
               Thang X. Vu and
               George K. Karagiannidis and
               Symeon Chatzinotas},
  title     = {{SCA-LLM}: Spectral-Attentive Channel Prediction with Large Language Models in {MIMO-OFDM}},
  journal   = {Under Review},
  year      = {2025}
}
```

## Acknowledgments

This project utilizes code from the [LLM4CP](https://github.com/liuboxun/LLM4CP) project for some of the baseline model comparisons. We are grateful to the authors of LLM4CP for making their work publicly available.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
