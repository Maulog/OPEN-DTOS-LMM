ACTIVITYNETCAPTION_VAL_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/activitycaption_val.jsonl',
    video_folder=r'/share_ssd/tianjirui/ActivityNet/video/v1-3/train_val',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/activity_caption',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_VAL_ACTIVITYNETCAPTION = dict(
    act_cap_val = dict(**ACTIVITYNETCAPTION_VAL_COMMON_CFG, version = 'v0'),
)