model_args = dict(
    type='dtos_lmm',
    version='v0',
    model_base=None,

    # checkpoint config
    model_name_or_path = '/share/tianjirui/Llama-3-VILA1.5-8b/',

    # load config
    dtype = 'torch.float16',
    load_8bit = False,
    load_4bit = False,    

    # model config
    mm_vision_select_feature = "siglip_cls_patch", # "patch", "cls_patch"
    mm_vision_select_layer = -2 ,
    # model_max_length=2048,

    # finetune config
    # freeze_backbone=True,

    # data process config
    sep_image_conv_front=False,
    image_token_len=196,
    mm_use_im_start_end=True,

    # mm config
    # mm_use_im_start_end=True,
    mm_use_im_patch_token = True,



    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ChatConvProcess'),
        target=dict(type='SegFormatProcess'),
        text=dict(type='ChatTextProcess'),
        image=dict(type='ChatImageProcessor'),
    ),

    conv_args=dict(
        conv_template='llama_3',
        transforms=None,
        tokenize_kwargs=dict(truncation_size=2048),
    ),
    image_aspect_ratio = 'resize',

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)