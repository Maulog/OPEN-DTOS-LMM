MEVIS_EXP_COMMON_CFG = dict(
    type='RVOSDataset',
    base_folder=r'/share_ssd/tianjirui/MeViS/example/', # 还没弄完
    template_file=r'{{fileDirname}}/template/prompts_seg_en.txt',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/mevis/',
    use_video=True,
    n_frms=15,
    sampling_type=None, # must provide
    dynamic_path=None,
)

DEFAULT_TRAIN_SEG_EXAMPLE = dict(
    mevis_example = dict(**MEVIS_EXP_COMMON_CFG, version = 'v0'),
)