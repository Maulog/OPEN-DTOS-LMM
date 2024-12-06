DIDEMO_VAL_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/didemo_val.jsonl',
    video_folder=r'/share_ssd/tianjirui/DiDeMo/videos/val_videos',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/didemo',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_VAL_DIDEMO = dict(
    didemo_val = dict(**DIDEMO_VAL_COMMON_CFG, version = 'v0'),
)