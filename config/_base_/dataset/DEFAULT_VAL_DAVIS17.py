DAVIS_VAL_COMMON_CFG = dict( # 待修改！！
    type='RVOSDataset',
    base_folder=r'/share_ssd/tianjirui/DAVIS/davis17_ytvos/valid/',
    template_file=r'{{fileDirname}}/template/prompts_seg_en.txt',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/davis/',
    use_video=True,
    n_frms=15,
    sampling_type=None, # must provide
    dynamic_path=None,
)

DEFAULT_VAL_DAVIS = dict(
    davis_val = dict(**DAVIS_VAL_COMMON_CFG, version = 'v0'),
)