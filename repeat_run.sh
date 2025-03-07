#!/bin/bash

# shellcheck disable=SC2054,SC2086

MODELS=(
  "VGG11"
  "VGG16"
  "VGG19"
  "ResNet50"
  "ResNet101"
  "ResNet152"
  "MobileNetV2"
  "MobeNetV3Small"
  "MobeNetV3Large"
  "MnasNet"
  "ConvNeXtTiny"
  "ConvNeXtBase"
  "RegNetX400MF"
  "RegNetX32GF"
  "RegNetY400MF"
  "RegNetY32GF"
)

optimizers=("SGD" "Adam" "RMSprop" "Adagrad" "AdamW")
optimizers=("Adam" "AdamW")


total_models="${#MODELS[@]}"
model_count=0

for MODEL in "${MODELS[@]}"; do
  model_count=$((model_count + 1))
  echo "Processing model: $MODEL ($model_count/$total_models)"

  for b in $(seq 10 40 530); do
    echo "  Batch size: $b"
    for optimize in "${optimizers[@]}"; do
      echo "    Optimizer: $optimize"
#      ID_xMem_LLM="${MODEL}-${b}-${optimize}"
      ID_xMem_LLM="tprofiler-${MODEL}-${optimize}-batch-${b}-Nvidia"
#      python tProfile.py -m "$MODEL" -b "$b" -o "$optimize" -u -g 8 -r "${ID_xMem_LLM}"
      NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python tProfile.py "$MODEL" -b "$b" -o "$optimize" -g 8 -r "${ID_xMem_LLM}" -c
      if [ $? -ne 0 ]; then
        echo "    ERROR: tProfile.py failed for $ID_xMem_LLM"
      fi
      sleep 5
    done
  done
done

echo "Script completed."
