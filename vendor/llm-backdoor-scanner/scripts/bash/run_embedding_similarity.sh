#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
cd "$(dirname "$0")/../.."

# Run embedding similarity analysis for various methods and seeds

METHODS=(fft lora qlora4b qlora8b)
SEEDS=(17 42 2025)

BASE_DIR="/mnt/lotrdata/backdoor_probes/results/leakage"
ENV_FILE="/home/blake/backdoor-probes/.env"

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir_name="Phi-4-mini-instruct--${method}--exp1--seed${seed}"
        dir_path="${BASE_DIR}/${dir_name}"
        
        # Check if directory exists
        if [[ ! -d "$dir_path" ]]; then
            echo "Skipping: Directory not found - $dir_path"
            continue
        fi
        
        # Check if similarity CSV already exists
        if ls "${dir_path}"/*_similarity.csv 1> /dev/null 2>&1; then
            echo "Skipping: Similarity CSV already exists in $dir_path"
            continue
        fi
        
        # Find the decoding sweep CSV file
        csv_file=$(ls "${dir_path}"/*--decoding_sweep--*.csv 2>/dev/null | grep -v "_similarity.csv" | head -n 1)
        
        if [[ -z "$csv_file" ]]; then
            echo "Skipping: No decoding sweep CSV found in $dir_path"
            continue
        fi
        
        echo "Processing: $csv_file"
        python /home/blake/backdoor-probes/scripts/analysis/leakage/embedding_similarity.py \
            --leakage_csv "$csv_file" \
            --env_file "$ENV_FILE"
        
        echo "Completed: $dir_name"
        echo "---"
    done
done

echo "All done!"
