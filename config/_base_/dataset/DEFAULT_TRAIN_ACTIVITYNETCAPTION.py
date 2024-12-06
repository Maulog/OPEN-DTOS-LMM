ACTIVITYNETCAPTION_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    # filename=r'{{fileDirname}}/../../../data/activitycaption_train_tree_max5.jsonl',
    filename=r'{{fileDirname}}/../../../data/activitycaption_train_tree.jsonl',
    video_folder=r'/share_ssd/tianjirui/ActivityNet/video/v1-3/train_val',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/activity_caption',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_TRAIN_ACTIVITYNETCAPTION = dict(
    act_cap = dict(**ACTIVITYNETCAPTION_COMMON_CFG, version = 'v0'),
)