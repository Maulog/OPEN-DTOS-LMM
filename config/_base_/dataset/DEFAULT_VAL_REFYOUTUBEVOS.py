REFYOUTUBEVOS_VAL_COMMON_CFG = dict(
    type='RVOSDataset',
    base_folder=r'/share_ssd1/tianjirui/ref_youtube_rvos/youtubervos/valid/',
    template_file=r'{{fileDirname}}/template/prompts_seg_en.txt',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/davis/',
    use_video=True,
    n_frms=15,
    sampling_type=None, # must provide
    dynamic_path=None,
)

DEFAULT_VAL_REFYOUTUBEVOS = dict(
    youtubervos_val = dict(**REFYOUTUBEVOS_VAL_COMMON_CFG, version = 'v0'),
)