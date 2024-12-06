_base_ = ['DEFAULT_TRAIN_LOCALIZATION_DATASET.py'] # 准备使用moment retrieval的数据集

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[  # 每个数据集构成当前数据集的配置（其中包括每个数据集的子集）
            # {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.example}}, # 用于测试数据集
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.charadessta}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.act_cap}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.didemo}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.qv}},
            # {{_base_.DEFAULT_TRAIN_TACOS.tacos_1}}, # 先不用
        ],
    ),
    validation=None,
    test=None,
    multitest=None,
    
    # compute_metric
    compute_metric=None, # TODO: add compute metric

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=2048,
    ),
    
    # dataset wrapper
    dataset_wrapper='conv_loc',
    
    # load video or feature 
    use_video = True, # feature is False
    
    # generate config
    gen_kwargs=dict(
        do_sample=True,
        output_scores=True, # 如果要输出beam_indices，需要设置为True
        output_hidden_states=True,
        return_dict_in_generate=True,
    ),

)
