_base_ = ['DEFAULT_LOCALIZATION_DATASET.py'] # 准备使用moment retrieval的数据集

data_args = dict(
    train=dict(
        type='ConcatDataset',
        cfgs=[  # 每个数据集构成当前数据集的配置（其中包括每个数据集的子集）
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.charadessta}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.act_cap}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.didemo}},
            {{_base_.DEFAULT_TRAIN_LOCALIZATION_DATASET.qv}},
        ],
    ),
    validation=dict(
        type='ConcatDataset',
        cfgs=[  # 每个数据集构成当前数据集的配置（其中包括每个数据集的子集）
            {{_base_.DEFAULT_VAL_LOCALIZATION_DATASET.charadessta_val}},
            {{_base_.DEFAULT_VAL_LOCALIZATION_DATASET.act_cap_val}},
            {{_base_.DEFAULT_VAL_LOCALIZATION_DATASET.didemo_val}},
            {{_base_.DEFAULT_VAL_LOCALIZATION_DATASET.qv_val}},
        ],
    ),
    test=dict(
        type='ConcatDataset',
        cfgs=[  # 每个数据集构成当前数据集的配置（其中包括每个数据集的子集）
            {{_base_.DEFAULT_TEST_LOCALIZATION_DATASET.charadessta_test}},
            {{_base_.DEFAULT_TEST_LOCALIZATION_DATASET.act_cap_test}},
            {{_base_.DEFAULT_TEST_LOCALIZATION_DATASET.didemo_test}},
            {{_base_.DEFAULT_TEST_LOCALIZATION_DATASET.qv_test}},
            # {{_base_.DEFAULT_TEST_LOCALIZATION_DATASET.qv_val}},
        ],
    ),
    multitest=None,

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # dataset wrapper
    dataset_wrapper='conv_loc',
    
    # load video or feature
    use_video = True,
    
    n_frms = 25,
    
    # generate config
    gen_kwargs=dict(
        do_sample=True,
        top_p = None,
        max_new_tokens=1024,
        num_beams=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
    ),
)