#!/bin/bash

# shellcheck disable=SC2054,SC2086

MODELS=(
#    "EleutherAI/gpt-neo-125M"
#    "facebook/opt-125m"
    "facebook/opt-350m"
#    "cerebras/Cerebras-GPT-111M"
    "microsoft/deberta-base"
#    "T5-small"
    "t5-base"
#    "distilbert/distilgpt2"
#    "openai-community/gpt2"
)

optimizers=("AdamW")


total_models="${#MODELS[@]}"
model_count=0

for MODEL in "${MODELS[@]}"; do
  model_count=$((model_count + 1))
  echo "Processing model: $MODEL ($model_count/$total_models)"

  for b in $(seq 5 10 65); do
    echo "  Batch size: $b"
    for optimize in "${optimizers[@]}"; do
      echo "    Optimizer: $optimize"
      for i in {1..5}; do
        python experiments_llm/evaluation.py --model "$MODEL" --device_id 0 --batch "${b}" --target_iteration 2 --optimiser "${optimize}"
        if [ $? -ne 0 ]; then
          echo "    ERROR: Failed to run evaluation for model $MODEL with batch size $b and optimizer $optimize"
        fi
      done
      sleep 5
    done
  done
done

echo "Script completed."
