#!/bin/bash

# run inference
export WANDB_JOB_TYPE=inference
python -u /full-model/fme/fme/fcn_training/inference/inference.py \
    /configmount/inference-config.yaml
