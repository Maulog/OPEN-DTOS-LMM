_base_ = [
    # 'DEFAULT_TRAIN_SEG_EXAMPLE.py',
    
    'DEFAULT_TEST_MEVIS.py',
    'DEFAULT_TEST_REFYOUTUBEVOS.py',
]

DEFAULT_TEST_SEGMENTATION_DATASET = dict(
    # **_base_.DEFAULT_TRAIN_SEG_EXAMPLE,
    
    **_base_.DEFAULT_TEST_MEVIS,
    ##### **_base_.DEFAULT_TEST_REFYOUTUBEVOS, # 不打算用，没有codalab测评
)