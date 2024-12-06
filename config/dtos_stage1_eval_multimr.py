import os
_base_ = ['_base_/dataset/dtos_stage1_eval.py', '_base_/model/dtos_loc.py', '_base_/train/eval.py']

model_args = dict(
    build_method = 'from_pretrained',
    loc_lora_path = '/share_ssd/tianjirui/dtos_output/stage1/exp102', # /checkpoint-50000
)

training_args = dict(
    output_dir = os.path.join(model_args['loc_lora_path'],'eval_top5'),  # '/share_ssd1/tianjirui/dtos_output/stage1/eval_exp64/',
    do_multi_predict = True,
)

data_args = dict(
    # load video or feature
    use_video = False,
    n_frms = 10,
    
    gen_kwargs=dict(
        do_sample=True,
        max_new_tokens=64, # 防止生成太长的重复token，不是根本的解决方法，只是为了评估而设置的
        no_repeat_ngram_size=10,
        # early_stopping=True,
        # top_k = 50,
        # top_p = 0.95,
        # num_beams=5,
        # temperature=1.2,
        # repetition_penalty=1.4, # 重复词惩罚
        # length_penalty=1.0,
    ),
)
