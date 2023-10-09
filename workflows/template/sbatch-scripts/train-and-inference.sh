#!/bin/bash

# run training
/full-model/fme/fme/fcn_training/run-train-and-inference.sh \
    /configmount/train-config.yaml \
    /configmount/inference-config.yaml \
    $SLURM_GPUS_PER_NODE
