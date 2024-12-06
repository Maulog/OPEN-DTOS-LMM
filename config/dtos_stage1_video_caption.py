_base_ = ['_base_/dataset/dtos_stage0.py', '_base_/model/dtos_lmm.py', '_base_/train/dtos_deepspeed.py']
# 数据集、模型、trainer
training_args = dict(
    num_train_epochs=0,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    type='dtos_lmm',
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=8192),
    )
    
)