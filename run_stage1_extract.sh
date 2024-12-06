#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
accelerate launch --num_processes 1 \
    --main_process_port 23788 \
    mllm/pipeline/extract_features.py \
    config/dtos_stage1_extract_feats.py \
    --output_dir ./output/stage1-extract-video