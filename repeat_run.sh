#!/bin/bash
MODEL="VGG16"
for b in $(seq 50 20 490); do
  python xProfile.py -m "$MODEL" -b "$b" -o "SGD" -u -g 8
  sleep 5
  python tProfile.py -m "$MODEL" -b "$b" -o "SGD" -u -g 8
  sleep 5
done