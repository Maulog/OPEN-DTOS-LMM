# 实例分割版本
import os
_base_ = ['_base_/dataset/dtos_stage2_example.py', '_base_/model/dtos_seg.py', '_base_/train/dtos_deepspeed.py']

model_args = dict(
    build_method = 'from_scratch', # from_pretrained, from_scratch
    model_base = '/share_ssd/tianjirui/dtos_output/stage1/exp95',
)

training_args = dict(
    logging_steps=1,
    save_steps=100000,
    save_total_limit=1, # 每次保存的最近保存数
    dataloader_num_workers=1,
    
    num_train_epochs=1,
    learning_rate=5e-5,
    overwrite_output_dir=True, # True 是从头训练  False 是继续训练
    output_dir='./output/stage2/exp_test/',
)

data_args = dict(
    # load video or feature 
    use_video = True, # feature is False
    n_frms = 7,
    sampling_type='Dynamic', # 'Dynamic', 'Uniform', 'Random', 'All'
    dynamic_path=os.path.join(model_args['model_base'], 'predict_rvos'),
)
