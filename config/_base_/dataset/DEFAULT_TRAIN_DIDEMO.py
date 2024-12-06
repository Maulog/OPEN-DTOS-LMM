DIDEMO_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    # filename=r'{{fileDirname}}/../../../data/didemo_train_tree_max5.jsonl',
    filename=r'{{fileDirname}}/../../../data/didemo_train_tree.jsonl',
    video_folder=r'/share_ssd/tianjirui/DiDeMo/videos/train_videos',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/didemo',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_TRAIN_DIDEMO = dict(
    didemo = dict(**DIDEMO_COMMON_CFG, version = 'v0'),
)