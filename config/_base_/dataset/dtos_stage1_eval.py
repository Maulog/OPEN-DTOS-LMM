_base_ = ['DEFAULT_TEST_LOCALIZATION_DATASET.py', 'DEFAULT_VAL_LOCALIZATION_DATASET.py'] # 准备使用moment retrieval的数据集

data_args = dict(
    train=None,
    validation=None,
    test=None, # 待修改
    multitest={
        **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_TEST_LOCALIZATION_DATASET.items()},
        **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_VAL_LOCALIZATION_DATASET.items() if k == 'qv_val'}, # 合并到multitest中
    },
    
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # dataset wrapper
    dataset_wrapper='conv_loc',
    
    # load video or feature
    use_video = False,
    
    # generate config
    gen_kwargs=dict(
        do_sample=True,
        output_scores=True, # 如果要输出beam_indices，需要设置为True
        output_hidden_states=True,
        return_dict_in_generate=True,
    ),
)