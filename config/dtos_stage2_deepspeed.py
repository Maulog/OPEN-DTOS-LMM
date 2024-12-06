# 实例分割版本
import os
_base_ = ['_base_/dataset/dtos_stage2.py', '_base_/model/dtos_seg.py', '_base_/train/dtos_deepspeed.py']

model_args = dict(
    build_method = 'from_scratch', # from_pretrained, from_scratch
    model_base = '/share_ssd/tianjirui/dtos_output/stage1/exp95', # 之后改exp95
)

training_args = dict(
    num_train_epochs=8,
    learning_rate=3e-5,
    dataloader_num_workers=2, # 这里试试1看看速度有没有变化 ！！！ # 使用2的内存消耗还是太大了
    overwrite_output_dir=True, # True 是从头训练  False 是继续训练
    output_dir='/share_ssd1/tianjirui/dtos_output/stage2/exp28/',
    save_steps=8000,
)

data_args = dict(
    # load video or feature 
    use_video = True, # feature is False
    n_frms = 7,
    sampling_type='Dynamic', # 'Dynamic', 'Uniform', 'Random', 'All'
    dynamic_path=os.path.join(model_args['model_base'], 'predict_rvos'),
)
