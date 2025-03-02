#!/bin/bash
MODELS = (
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
#MODELS=("ConvNeXtTiny" "ResNet50" "EfficientNetB0")
optimizers = ("SGD", "Adam", "RMSprop", "Adagrad", "AdamW")
for MODEL in "${MODELS[@]}"; do
  for b in $(seq 10 40 530); do
    ID_xMem_LLM="${MODEL}-${b}-LLM"
    python tProfile.py -m "$MODEL" -b "$b" -o "SGD" -u -g 8 -r "${ID_xMem_LLM}"
    sleep 5
  done
done
