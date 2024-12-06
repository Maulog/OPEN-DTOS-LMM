export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL 
    
accelerate launch \
    --config_file config/_base_/accelerate/zero2.yaml \
    --num_processes 4 \
    --main_process_port 23820 mllm/pipeline/finetune.py \
    config/dtos_stage1_deepspeed.py
