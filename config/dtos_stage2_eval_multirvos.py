import os
_base_ = ['_base_/dataset/dtos_stage2_eval.py', '_base_/model/dtos_seg.py', '_base_/train/eval.py']

model_args = dict(
    build_method = 'from_pretrained', # from_pretrained, from_scratch
    model_base = '/share_ssd/tianjirui/dtos_output/stage1/exp95',                        # lora1
    seg_lora_path = '/share_ssd1/tianjirui/dtos_output/stage2/exp13',   # lora2   # /checkpoint-6000
)

training_args = dict(
    output_dir = os.path.join(model_args['seg_lora_path'],'eval_1tgt_method3_nms07_box1'),  # 'eval_1tgt_method3_nms07',
    do_multi_predict = True,
    eval_accumulation_steps = 1,
    dataloader_num_workers = 1,
)

data_args = dict(
    # load video or feature 
    use_video = True, # feature is False
    n_frms = 7,
    sampling_type='Dynamic', # 'Dynamic', 'Uniform', 'Random', 'All', 'Dynamic_and_Uniform'
    dynamic_path=os.path.join(model_args['model_base'], 'predict_rvos'),
    
    gen_kwargs=dict(
        do_sample=True,
        max_new_tokens=64, # 防止生成太长的重复token，不是根本的解决方法，只是为了评估而设置的
        # no_repeat_ngram_size=10,
        # early_stopping=True,
        # top_k = 50,
        # top_p = 0.95,
        # num_beams=5,
        # temperature=1.2,
        # repetition_penalty=1.4, # 重复词惩罚
        # length_penalty=1.0,
    ),
)
