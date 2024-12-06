EXAMPLES_COMMON_TEST_CFG = dict(
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/example_test_max5.jsonl',
    video_folder=r'/share_ssd/tianjirui/QVHighlishts/videos',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/qvhighlights',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_TEST_EXAMPLES = dict(
    example = dict(**EXAMPLES_COMMON_TEST_CFG, version = 'v0'),
)