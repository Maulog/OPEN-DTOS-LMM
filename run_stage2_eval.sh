export CUDA_VISIBLE_DEVICES=2,3
    
accelerate launch \
    --config_file config/_base_/accelerate/no_ds.yaml \
    --num_processes 2 \
    --main_process_port 23817 mllm/pipeline/finetune.py \
    config/dtos_stage2_eval_multirvos.py