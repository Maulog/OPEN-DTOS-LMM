_base_ = ['_base_/dataset/dtos_stage1.py', '_base_/model/dtos_loc.py', '_base_/train/dtos_deepspeed.py']
# 数据集、模型、trainer
data_args = dict(
    train=dict(
        type='ConcatDataset',
        cfgs=[  # 每个数据集构成当前数据集的配置（其中包括每个数据集的子集）
            # {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.example}}, # 用于测试数据集
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.charadessta}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.act_cap}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.didemo}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.qv}},
        ],
    ),
    use_video = False,
    n_frms = 10,
    
)

training_args = dict( # 常用调参的位置
    num_train_epochs=2,
    learning_rate=5e-5,
    output_dir='./output/stage1/exp_test/',
    overwrite_output_dir=True, # 这用true就是从头训练，false就是继续训练
    
    # train logging
    logging_steps=1,
    save_strategy='steps',
    save_steps=10000, #3000,
    save_total_limit=1,#10, # 每次保存的最近保存数
    
    dataloader_num_workers=4, # 4
)

model_args = dict(
    type='dtos_loc',
    load_lora = True,                 # debug use False
    lora_path = None,                 
    lora_cfg = dict(
        lora_r = 64,   # 64           # lora attention demension
        lora_alpha = 128,             # The alpha parameter for Lora scaling.
        lora_dropout = 0.05,
        lora_bias = "none",
        task_type = "CAUSAL_LM",
    ),
    
)
