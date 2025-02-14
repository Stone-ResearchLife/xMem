#!/bin/bash
MODEL="ConvNeXtTiny"
for b in $(seq 10 40 530); do
  ID_xMem="${MODEL}-${b}-xMem"
  ID_xMem_LLM="${MODEL}-${b}-LLM"
  ID_xMem_LLM_CUDA="${MODEL}-${b}-CUDA"
  python xProfile.py -m "$MODEL" -b "$b" -o "SGD" -u -g 8 -r "${ID_xMem}"
  sleep 5
  python tProfile.py -m "$MODEL" -b "$b" -o "SGD" -u -g 8 -r "${ID_xMem_LLM}"
  sleep 5
  python tProfile.py -m "$MODEL" -b "$b" -o "SGD" -c -g 8 -r "${ID_xMem_LLM_CUDA}"
  sleep 5
done