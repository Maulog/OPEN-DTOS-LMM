QVHIGHLIGHTS_VAL_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/qvhighlights_val.jsonl',
    video_folder=r'/share_ssd/tianjirui/QVHighlishts/videos',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/qvhighlights',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_VAL_QVHIGHLIGHTS = dict(
    qv_val = dict(**QVHIGHLIGHTS_VAL_COMMON_CFG, version = 'v0'),
)