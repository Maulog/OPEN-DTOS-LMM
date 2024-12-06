#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
accelerate launch --config_file deepspeed_zero2.yaml\
    --num_processes 1 \
    --main_process_port 23786 mllm/models/llava/eval/run_vila.py \
    --model-path /share/tianjirui/Llama-3-VILA1.5-8b \
    --conv-mode llama_3 \
    --query "<video>\n Please describe this video." \
    --video-file "demo/musical.mov"