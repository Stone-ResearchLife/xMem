#!/bin/bash

# shellcheck disable=SC2054,SC2086

MODELS=(
    "EleutherAI/gpt-neo-125M"
    "facebook/opt-125m"
    "cerebras/Cerebras-GPT-111M"
    "t5-base"
    "microsoft/deberta-base"
)

optimizers=("SGD" "Adam" "RMSprop" "Adagrad" "AdamW")
optimizers=("SGD" "AdamW")


total_models="${#MODELS[@]}"
model_count=0

for MODEL in "${MODELS[@]}"; do
  model_count=$((model_count + 1))
  echo "Processing model: $MODEL ($model_count/$total_models)"

  for b in $(seq 10 5 20); do
    echo "  Batch size: $b"
    for optimize in "${optimizers[@]}"; do
      echo "    Optimizer: $optimize"
      for i in {1..5}; do
        python experiments_llm/evaluation.py --model "$MODEL" --device_id 1 --batch "${b}" --target_iteration 2 --optimiser "${optimize}"
        if [ $? -ne 0 ]; then
          echo "    ERROR: Failed to run evaluation for model $MODEL with batch size $b and optimizer $optimize"
        fi
      done
      sleep 5
    done
  done
done

echo "Script completed."
