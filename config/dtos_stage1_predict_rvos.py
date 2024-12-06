import os
_base_ = ['_base_/dataset/dtos_stage1_predict.py', '_base_/model/dtos_loc.py', '_base_/train/eval.py']

model_args = dict(
    build_method = 'from_pretrained',
    loc_lora_path = '/share_ssd/tianjirui/dtos_output/stage1/exp95', # checkpoint-70000
)

training_args = dict(
    output_dir = os.path.join(model_args['loc_lora_path'],'predict_rvos'),  # '/share_ssd1/tianjirui/dtos_output/stage1/eval_exp64/',
    do_multi_predict = True,
)

data_args = dict(
    # load video or feature
    use_video = True,
    n_frms = 10, # 必须和训练时匹配
    
    # 因为这里是使用整个视频信息进行预测，所以用Uniform
    sampling_type='Uniform', # 'Dynamic', 'Uniform', 'Random', 'All'
    dynamic_path=None, # os.path.join(training_args['output_dir'], 'mevis/exp83_top1.pth'), # 输出路径
    
    gen_kwargs=dict(
        do_sample=True,
        max_new_tokens=64, # 防止生成太长的重复token，不是根本的解决方法，只是为了评估而设置的
        no_repeat_ngram_size=10,
    ),
)
