CHARADESSTA_VAL_COMMON_CFG = dict(
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/charadessta_val.jsonl',
    video_folder=r'/share_ssd/tianjirui/Charades-STA/Charades_v1',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/charadessta',
    template_file=r'{{fileDirname}}/template/prompts_en.txt',
    use_video=False,
    n_frms=15,
)

DEFAULT_VAL_CHARADESSTA = dict(
    charadessta_val = dict(**CHARADESSTA_VAL_COMMON_CFG, version = 'v0'),
)