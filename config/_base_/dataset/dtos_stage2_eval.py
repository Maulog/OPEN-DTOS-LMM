_base_ = ['DEFAULT_TEST_SEGMENTATION_DATASET.py', 'DEFAULT_VAL_SEGMENTATION_DATASET.py'] # 准备使用moment retrieval的数据集

data_args = dict(
    train=None,
    validation=None,
    test=None,
    multitest={
        **{k: {'cfg': v, 'compute_metric': dict(type='RVOSComputeMetrics')} for k, v in _base_.DEFAULT_VAL_SEGMENTATION_DATASET.items()},
        **{k: {'cfg': v, 'compute_metric': dict(type='RVOSComputeMetrics')} for k, v in _base_.DEFAULT_TEST_SEGMENTATION_DATASET.items()},
    },

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # dataset wrapper
    dataset_wrapper='conv_seg',
    
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