#!/bin/bash

# Define parameters
velocities=(0 15 30 50 60)
snrs=(0 5 10 15)

# Define explicitly specified output directories
declare -A output_dirs
# NMSE output directories
output_dirs["nmse_velocity"]="ablation_experiment/figures/UMa/nmse_vs_velocity"
output_dirs["nmse_snr"]="ablation_experiment/figures/UMa/nmse_vs_snr"
output_dirs["nmse_dataframe"]="ablation_experiment/figures/UMa/nmse_vs_dataframe"

# Common parameters
labels=("w/o SCA" "w/o GPT-2" "Ours(SCA-LLM)")
results=(
    "ablation_experiment/results/GPT2_sl24_pl6_20250407_222239_eval_20250408_000003.json"
    "ablation_experiment/results/SPAGPT2_wo_GPT2_sl24_pl6_20250407_213311_eval_20250407_235831.json"
    "ablation_experiment/results/SPAGPT2_sl24_pl6_20250407_092052_eval_20250407_114319.json"
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