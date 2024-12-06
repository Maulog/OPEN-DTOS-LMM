_base_ = ['DEFAULT_SEGMENTATION_DATASET.py'] # 准备使用moment retrieval的数据集

data_args = dict(
    train=None,
    validation=None,
    test=None, # 待修改
    multitest={
        # **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_TRAIN_SEG_EXAMPLE.items()},
        
        **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_TRAIN_SEGMENTATION_DATASET.items()},
        **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_VAL_SEGMENTATION_DATASET.items()},
        **{k: {'cfg': v, 'compute_metric': dict(type='MRComputeMetrics')} for k, v in _base_.DEFAULT_TEST_SEGMENTATION_DATASET.items()},
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
    use_video = True,
    
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