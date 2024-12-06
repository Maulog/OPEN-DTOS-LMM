REFYOUTUBEVOS_TEST_COMMON_CFG = dict(
    type='RVOSDataset',
    base_folder=r'/share_ssd1/tianjirui/ref_youtube_rvos/youtubervos/test/',
    template_file=r'{{fileDirname}}/template/prompts_seg_en.txt',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/davis/',
    use_video=True,
    n_frms=15,
    sampling_type=None, # must provide
    dynamic_path=None,
)

DEFAULT_TEST_REFYOUTUBEVOS = dict(
    youtubervos_test = dict(**REFYOUTUBEVOS_TEST_COMMON_CFG, version = 'v0'),
)