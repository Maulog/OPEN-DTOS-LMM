TACOS_COMMON_CFG = dict( # 暂时不用先不管
    type='MomentRetrievalTreeDataset',
    filename=r'{{fileDirname}}/../../../data/TACoS_train.jsonl',
    video_folder=r'/share_ssd/tianjirui/Charades_STA/Charades_v1',
    feat_folder=r'/share_ssd1/tianjirui/feats/stage1/tacos',
    template_file=r'{{fileDirname}}/template/MR.json',
)

DEFAULT_TRAIN_TACOS = dict(
    tacos_1 = dict(**TACOS_COMMON_CFG, version = 'v0'),
    tacos_2 = dict(**TACOS_COMMON_CFG, version = 'v1'), # 改标注位置？
)