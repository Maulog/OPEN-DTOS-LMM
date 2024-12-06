export CUDA_VISIBLE_DEVICES=0,1,2,3
    
accelerate launch \
    --config_file config/_base_/accelerate/no_ds.yaml \
    --num_processes 4 \
    --main_process_port 23813 mllm/pipeline/finetune.py \
    config/dtos_stage1_eval_multimr.py