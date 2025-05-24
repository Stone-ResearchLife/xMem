#!/usr/bin/env bash

NUMBER_OF_RUNS=5
MODEL=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
	"Qwen/Qwen3-4B"
	"meta-llama/Llama-3.2-3B-Instruct"
)
OPTIMIZERS=("Adafactor" "AdamW" "SGD", "Adam")
BATCH=1


for OPTIMIZER in "${OPTIMIZERS[@]}"
do
  for i in $(seq 1 $NUMBER_OF_RUNS)
  do
      echo "Run #$i of $NUMBER_OF_RUNS"
      python exp/run_in_colab.py --model ${MODEL} --bs ${BATCH} --optimizer ${OPTIMIZER}

      # If it's not the last run and DELAY_SECONDS is greater than 0, then sleep.
      if [ "$i" -lt "$NUMBER_OF_RUNS" ] && [ "$DELAY_SECONDS" -gt 0 ]; then
          echo "Waiting for $DELAY_SECONDS seconds before the next run..."
          sleep "$DELAY_SECONDS"
      fi
  done
done