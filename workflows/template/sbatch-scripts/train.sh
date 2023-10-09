#!/bin/bash

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES \
    --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d \
    --rdzv_endpoint $ENDPOINT_HOST:29500 \
    /full-model/fme/fme/fcn_training/train.py \
    --yaml_config /configmount/train-config.yaml
