model_args = dict(
    type='dtos_seg',
    version='v0',
    model_base=None,
    seg_dim = 2,
    
    build_method = 'from_pretrained', # 'from_scratch', 'from_pretrained'

    # checkpoint config
    model_name_or_path = '/share/tianjirui/Llama-3-VILA1.5-8b/',
    # sam2 config
    sam2_cfg = "sam2_hiera_l.yaml", # 这里必须相对路径
    sam2_path = "/share_ssd/tianjirui/SAM2/checkpoints/sam2_hiera_large.pt",

    # load config llm 这里待修改,还没传参
    dtype = 'torch.float16',
    load_8bit = False,
    load_4bit = False, 

    # model config
    mm_vision_select_feature = "siglip_cls_patch", # "patch", "cls_patch"
    mm_vision_select_layer = -2 ,
    
    # lora config
    load_lora = True,                # debug use False
    loc_lora_path = None,                 
    lora_cfg = dict(
        lora_r = 64,                  # lora attention demension
        lora_alpha = 128,             # The alpha parameter for Lora scaling.
        lora_dropout = 0.05,
        lora_bias = "none",
        task_type = "CAUSAL_LM",
    ),
    
    # matcher config
    cost_class=1, 
    cost_bbox=1, 
    cost_giou=1, 
    span_loss_type="l1",

    # add special token
    add_special_tokens=True,

    # data process config
    sep_image_conv_front=False,
    image_token_len=196,
    video_token_len=196,
    
    # mm config
    mm_use_im_start_end=True,
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
        transforms=None, # dict(type='resize'), # 这里可能没什么用
        tokenize_kwargs=dict(truncation_size=8192),
    ),
    image_aspect_ratio = 'resize',

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)
