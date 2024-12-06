export CUDA_VISIBLE_DEVICES=4,5
    
accelerate launch \
    --config_file config/_base_/accelerate/no_ds.yaml \
    --num_processes 2 \
    --main_process_port 23815 mllm/pipeline/finetune.py \
    config/dtos_stage1_predict_rvos.py