_base_ = [
    'DEFAULT_VAL_MEVIS.py',
    'DEFAULT_VAL_DAVIS17.py',
    'DEFAULT_VAL_REFYOUTUBEVOS.py',
]

DEFAULT_VAL_SEGMENTATION_DATASET = dict(
    **_base_.DEFAULT_VAL_MEVIS,
    **_base_.DEFAULT_VAL_DAVIS,
    **_base_.DEFAULT_VAL_REFYOUTUBEVOS, # 从这验证
)