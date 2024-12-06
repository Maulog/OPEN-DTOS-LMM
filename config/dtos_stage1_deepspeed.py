_base_ = ['_base_/dataset/dtos_stage1.py', '_base_/model/dtos_loc.py', '_base_/train/dtos_deepspeed.py']
# 数据集、模型、trainer
training_args = dict( # 常用调参的位置
    num_train_epochs=8,
    learning_rate=5e-5,
    overwrite_output_dir=False, # True 是从头训练  False 是继续训练
    output_dir='/share_ssd/tianjirui/dtos_output/stage1/exp102/',
)

data_args = dict(
    # load video or feature 
    use_video = False, # feature is False
    n_frms = 10,
)
