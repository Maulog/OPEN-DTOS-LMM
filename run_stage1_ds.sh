export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_LEVEL=NVL 
    
deepspeed --num_gpus 2 \
    mllm/pipeline/finetune.py \
    config/dtos_stage1_deepspeed.py \
    --deepspeed config/_base_/train/zero1.json \
    
