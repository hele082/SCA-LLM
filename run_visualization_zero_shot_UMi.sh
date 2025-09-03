#!/bin/bash

# Define parameters
velocities=(25 50)
snrs=(0 5 10 15)

# Define explicitly specified output directories
declare -A output_dirs
# NMSE output directories
output_dirs["nmse_velocity"]="Figures/zero_shot_UMi/nmse_vs_velocity"
output_dirs["nmse_snr"]="Figures/zero_shot_UMi/nmse_vs_snr"
output_dirs["nmse_dataframe"]="Figures/zero_shot_UMi/nmse_vs_dataframe"

# Common parameters
labels=("RNN" "LSTM" "GRU" "Transformer" "LLM4CP" "Ours(SCA-LLM)")
results=(
    "Results/zero_shot_UMi/RNN_sl24_pl6_20250407_055018_eval_20250407_143236.json"
    "Results/zero_shot_UMi/LSTM_sl24_pl6_20250407_050726_eval_20250407_143243.json"
    "Results/zero_shot_UMi/GRU_sl24_pl6_20250407_054721_eval_20250407_143315.json"
    "Results/zero_shot_UMi/Informer_sl24_pl6_20250407_124130_eval_20250407_142914.json"
    "Results/zero_shot_UMi/GPT4CP_sl24_pl6_20250407_170951_eval_20250407_214340.json"
    "Results/zero_shot_UMi/SPAGPT2_sl24_pl6_20250407_092052_eval_20250407_140320.json"
)

# 1. NMSE vs Velocity (fixing SNR)
echo "Running visualization for NMSE vs velocity"
for s in "${snrs[@]}"; do
    output_dir="${output_dirs["nmse_velocity"]}/snr${s}"
    mkdir -p "$output_dir"
    
    python result_visualization.py \
        --labels "${labels[@]}" \
        --results "${results[@]}" \
        --metric "nmse" \
        --output_dir "$output_dir" \
        --plot_type "velocity" \
        --snr "$s"
done

# 2. NMSE vs SNR (fixing velocity)
echo "Running visualization for NMSE vs SNR"
for v in "${velocities[@]}"; do
    output_dir="${output_dirs["nmse_snr"]}/v${v}"
    mkdir -p "$output_dir"
    
    python result_visualization.py \
        --labels "${labels[@]}" \
        --results "${results[@]}" \
        --metric "nmse" \
        --output_dir "$output_dir" \
        --plot_type "snr" \
        --velocity "$v"
done

# 3. NMSE vs Dataframe (for each velocity and SNR combination)
echo "Running visualization for NMSE vs dataframe"
for v in "${velocities[@]}"; do
    for s in "${snrs[@]}"; do
        output_dir="${output_dirs["nmse_dataframe"]}/v${v}_snr${s}"
        mkdir -p "$output_dir"
        
        python result_visualization.py \
            --labels "${labels[@]}" \
            --results "${results[@]}" \
            --metric "nmse" \
            --output_dir "$output_dir" \
            --velocity "$v" \
            --snr "$s" \
            --plot_type "step"
    done
done

echo "All visualizations completed!" 