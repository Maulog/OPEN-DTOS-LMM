'''
    该文件中主要包含三个部分，分别加载不同模型结构的预训练参数
    lmm(llm_pretrained + vision_tower + projector)
    loc(lmm + lora1)
    seg(loc + sam + lora2)
'''
import os
import warnings
import shutil
import json
from typing import Dict, Any, Tuple

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    LlamaConfig,
    
)
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from mllm.engine.registry import LOAD_PRETRAINED
from mllm.models.sam2.build_sam import build_sam2, build_sam2_video_predictor
from mllm.models.sam2.sam2_image_predictor import SAM2ImagePredictor

from ..dtos.dtos_seg import DtosForSegLM
from ..dtos.dtos_base import DtosLmmForCausalLM, DtosConfig
from ..dtos.dtos_loc import DtosForLocLM
from ..dtos.dtos_debug import DtosForDebug

from mllm.models import *
from mllm.utils.common import (
    is_mm_model, 
    smart_resize_dtos_loc, 
    smart_resize_dtos_seg,
    smart_tokenizer_and_partial_embedding_resize, 
    load_state_dict_with_warning
)
from mllm.config.constants import *

from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
# from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import smart_tokenizer_and_embedding_resize


from llava.model import *
from llava.model.utils import is_mm_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.model.language_model.llava_llama import LlavaLlamaModel



PREPROCESSOR = Dict[str, Any]

def load_lora_and_other(model, base_path, model_name):
    non_lora_trainables_path = os.path.join(base_path, model_name, model_name+'.bin') # 非lora的保存
    assert os.path.exists(non_lora_trainables_path), f"Cannot find {model_name}.bin"
    non_lora_trainables = torch.load(non_lora_trainables_path)
    
    # 处理保存中命名的部分
    # non_lora_trainables = {k.replace('orig_emb.', ''): v for k, v in non_lora_trainables.items() if 'orig_emb.' in k }
    non_lora_trainables = {k[17:]: v for k, v in non_lora_trainables.items() if k.startswith('base_model.model.')}
    keys_to_change = list(non_lora_trainables.keys())
    for k in keys_to_change:
        if 'orig_emb.' in k:
            new_key = k.replace('orig_emb.', '')
            non_lora_trainables[new_key] = non_lora_trainables.pop(k)
        
    load_state_dict_with_warning(model, non_lora_trainables) # 是能检查要添加的键是否在模型中
    model.load_state_dict(non_lora_trainables, strict=False) # 少一个没加载进去会报错
    model = PeftModel.from_pretrained(model, base_path)
    
    print('\nmerge and unload ...\n')
    model = model.merge_and_unload()
    
    return model



@LOAD_PRETRAINED.register_module()
def load_pretrained_dtos_lmm(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]: 
    # 一阶段，以加载vila为例
    config, kwargs = prepare_model_args(model_args, **kwargs)
    
    model = DtosLmmForCausalLM(config=config, low_cpu_mem_usage=True, **kwargs)
    tokenizer = model.tokenizer
    
    tokenizer, model, image_processor, context_len = model_post_process(
        model, 
        model_args.model_name_or_path, 
        tokenizer, 
    )


    # 此处将image_processor、tokenizer等包装成preprocessor
    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    
    model.requires_grad_(False) # 冻结全参数
    
    
    
    
    import json
    os.makedirs(training_args.output_dir, exist_ok=True)
    param_path = os.path.join(training_args.output_dir, "param.json") 
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open(param_path, "w"))
    return model, preprocessor
    
    
@LOAD_PRETRAINED.register_module()
def load_pretrained_dtos_loc(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]: 
    config, kwargs = prepare_model_args(model_args, **kwargs)
    
    
    additional_special_tokens = [
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
        DEFAULT_VIDEO_TOKEN,
        VIDEO_PLACEHOLDER,
        DEFAULT_MOMENT_TOKEN,
        DEFAULT_MO_START_TOKEN,
        DEFAULT_MO_END_TOKEN,
        DEFAULT_VI_START_TOKEN,
        DEFAULT_VI_END_TOKEN,
    ]
    
    if model_args.build_method == "from_scratch":
        model = DtosForLocLM(config=config, low_cpu_mem_usage=True, **kwargs)
        model.requires_grad_(False) # 冻结所有参数
        tokenizer = model.tokenizer
        
        tokenizer, model, image_processor, context_len = model_post_process( # 用于设置设备和数据类型
            model, 
            model_args.model_name_or_path, 
            tokenizer
        )
        
        
        if model_args.load_lora:
            # from peft import LoraConfig, get_peft_model
            lora_cfg = model_args.lora_cfg
            lora_config = LoraConfig( # lora初始化
                r = lora_cfg.lora_r,
                lora_alpha = lora_cfg.lora_alpha,
                target_modules = find_all_linear_names(model.llm), # 仅给大模型添加lora
                lora_dropout = lora_cfg.lora_dropout,
                bias = lora_cfg.lora_bias,
                task_type = lora_cfg.task_type,
            )
            model = get_peft_model(model, lora_config)
        
        if model_args.add_special_tokens: # 添加特殊token
            smart_tokenizer_and_partial_embedding_resize(
                {'additional_special_tokens': additional_special_tokens},
                tokenizer,
                model,
            )
            # model.get_input_embeddings().requires_grad_(True) # 维度是词表的长度, 此处冻住，只训练新增加的参数，降低显存占用
            model.get_rec_token_projector().init_train_requires_grad() 
            # model.get_rec_token_classifier().init_train_requires_grad() 
            model.get_output_embeddings().requires_grad_(True)
            model.rec_decoder.requires_grad_(True)
            # model.rec_score_head.requires_grad_(True) # 问题在这里，暂时没想到怎么解决（双卡测试同样是一个有一个没有）
            # model.rec_encoder.requires_grad_(True)
            
            
            
    elif model_args.build_method == "from_pretrained": #
        model = DtosForLocLM(config=config, low_cpu_mem_usage=True, **kwargs) # 这不使用low_cpu_mem_usage和device_map,因为stage3限制
        model.requires_grad_(False) # 冻结所有参数
        
        tokenizer = AutoTokenizer.from_pretrained(model_args.loc_lora_path)
        model.tokenizer = tokenizer
        
        tokenizer, model, image_processor, context_len = model_post_process(
            model, 
            model_args.model_name_or_path, 
            tokenizer
        )
        
        smart_resize_dtos_loc(model, tokenizer) # 用于直接调整模型的输入输出层，用于评估
        
        print('load stage1 param ...')
        model = load_lora_and_other(model, model_args.loc_lora_path, model_name='dtos_loc')
        
        model.requires_grad_(False) # 冻结所有参数
        
            
            
    # 此处将image_processor、tokenizer等包装成preprocessor
    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            video_token_len=model_args.video_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    
    import json
    os.makedirs(training_args.output_dir, exist_ok=True)
    param_path = os.path.join(training_args.output_dir, "param.json")
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open(param_path, "w"))
    return model, preprocessor




@LOAD_PRETRAINED.register_module()
def load_pretrained_dtos_seg(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]: # 没写完
    config, kwargs = prepare_model_args(model_args, **kwargs)
    
    additional_special_tokens = [
        ### add new token ###
        DEFAULT_BOX_TOKEN,
        # DEFAULT_POS_TOKEN,
        # DEFAULT_NEG_TOKEN,
        
        DEFAULT_TGT_START_TOKEN,
        DEFAULT_TGT_END_TOKEN,
        
        DEFAULT_MOM_START_TOKEN,
        DEFAULT_MOM_END_TOKEN,
        
        DEFAULT_TGT_FIRST_TOKEN,
        DEFAULT_TGT_SECOND_TOKEN,
        DEFAULT_TGT_THIRD_TOKEN,
        DEFAULT_TGT_FOURTH_TOKEN,
        DEFAULT_TGT_FIFTH_TOKEN,
        DEFAULT_TGT_SIXTH_TOKEN,
        DEFAULT_TGT_SEVENTH_TOKEN,
        DEFAULT_TGT_EIGHTH_TOKEN,
        DEFAULT_TGT_NINTH_TOKEN,
        DEFAULT_TGT_TENTH_TOKEN,
    ]
    
    def build_sam2_predictor(device): # sam2 must be build in outside of the model(because of the dtype)
        # use bfloat16 for the entire notebook
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        sam2_model = build_sam2(config.sam2_cfg, config.sam2_path, device = device)
        sam2_img_predictor = SAM2ImagePredictor(sam2_model)
        sam2_vid_predictor = build_sam2_video_predictor(config.sam2_cfg, config.sam2_path) # 这里加载fp32的，但是前向会混合精度变成bf16
        
        return sam2_img_predictor, sam2_vid_predictor
    
    # load stage1 parameters
    model = DtosForSegLM(config=config, low_cpu_mem_usage=True, **kwargs)
    # model.sam2_img_predictor, model.sam2_vid_predictor = build_sam2_predictor(model.device)
    model.requires_grad_(False) # 冻结所有参数
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_base)
    model.tokenizer = tokenizer
    
    # 不在该函数中加载lora
    tokenizer, model, image_processor, context_len = model_post_process( # 用于设置设备和数据类型
        model, 
        model_args.model_name_or_path, 
        tokenizer
    )
    
    smart_resize_dtos_loc(model, tokenizer)
    
    ''' # 这里用于控制是否添加第一阶段的lora信息，因为tokenizer中有共用的，所以不改前面的地方
    print('load stage1 param ...')
    model = load_lora_and_other(model, model_args.model_base, model_name='dtos_loc')
    '''
    
    
    if model_args.build_method == "from_scratch":
        model.sam2_img_predictor, _ = build_sam2_predictor(model.device)
        model.requires_grad_(False) # 冻结所有参数
        
        if model_args.load_lora:
            lora_cfg = model_args.lora_cfg
            lora_config = LoraConfig( # lora初始化
                r = lora_cfg.lora_r,
                lora_alpha = lora_cfg.lora_alpha,
                target_modules = find_all_linear_names(model.llm), # 仅给大模型添加lora
                lora_dropout = lora_cfg.lora_dropout,
                bias = lora_cfg.lora_bias,
                task_type = lora_cfg.task_type,
            )
            model = get_peft_model(model, lora_config)
        
        if model_args.add_special_tokens: # 添加特殊token
            smart_tokenizer_and_partial_embedding_resize(
                {'additional_special_tokens': additional_special_tokens},
                tokenizer,
                model,
            )
            model.get_seg_token_projector().init_train_requires_grad() 
            model.get_output_embeddings().requires_grad_(True)
            model.box_decoder.requires_grad_(True)
            # model.seg_score_head.requires_grad_(True) # 尝试注释
            
            
            
    elif model_args.build_method == "from_pretrained":
        _, model.sam2_vid_predictor = build_sam2_predictor(model.device)
        model.requires_grad_(False) # 冻结所有参数
        
        tokenizer = AutoTokenizer.from_pretrained(model_args.seg_lora_path)
        model.tokenizer = tokenizer
        
        tokenizer, model, image_processor, context_len = model_post_process( # 用于设置设备和数据类型
            model, 
            model_args.model_name_or_path, 
            tokenizer
        )
        
        smart_resize_dtos_seg(model, tokenizer)
        
        print('load stage2 param ...')
        model = load_lora_and_other(model, model_args.seg_lora_path, model_name='dtos_seg')
        
        model.requires_grad_(False) # 冻结所有参数
            
    # 此处将image_processor、tokenizer等包装成preprocessor
    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            video_token_len=model_args.video_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    
    import json
    os.makedirs(training_args.output_dir, exist_ok=True)
    param_path = os.path.join(training_args.output_dir, "param.json")
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open(param_path, "w"))
    return model, preprocessor




@LOAD_PRETRAINED.register_module()
def load_pretrained_dtos_debug(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]: 
    config, kwargs = prepare_model_args(model_args, **kwargs)
    
    model = DtosForDebug(config=config, low_cpu_mem_usage=True, **kwargs)
    tokenizer = model.tokenizer
    
    tokenizer, model, image_processor, context_len = model_post_process(
        model, 
        model_args.model_name_or_path, 
        tokenizer, 
    )


    # 此处将image_processor、tokenizer等包装成preprocessor
    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            video_token_len=model_args.video_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    
    model.requires_grad_(False) # 冻结全参数
    model.rec_decoder.requires_grad_(True)
    
    
    
    import json
    os.makedirs(training_args.output_dir, exist_ok=True)
    param_path = os.path.join(training_args.output_dir, "param.json") 
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open(param_path, "w"))
    return model, preprocessor




    
def prepare_model_args(
    model_args,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if model_args.load_8bit:
        kwargs["load_in_8bit"] = True
    elif model_args.load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = eval(model_args.dtype)
    
    model_path = model_args.model_name_or_path
    
    # 重新加载部分config
    config = DtosConfig.from_pretrained(model_path)
    # init config
    config.resume_path = model_path
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_use_im_start_end = model_args.mm_use_im_start_end # getattr(model.config, "mm_use_im_start_end", False)
    config.mm_use_im_patch_token = model_args.mm_use_im_patch_token # getattr(model.config, "mm_use_im_patch_token", True)
    config.rec_dim = getattr(model_args, "rec_dim", 2)
    
    config.cost_class = getattr(model_args, "cost_class", 1)
    config.cost_span = getattr(model_args, "cost_span", 1)
    config.cost_bbox = getattr(model_args, "cost_bbox", 1)
    config.cost_giou = getattr(model_args, "cost_giou", 1)
    config.span_loss_type = getattr(model_args, "span_loss_type", "l1")
    
    config.seg_dim = getattr(model_args, "seg_dim", 2)
    config.sam2_cfg = getattr(model_args, "sam2_cfg", None)
    config.sam2_path = getattr(model_args, "sam2_path", None)
    
    
    
    prepare_config_for_eval(config, kwargs)
    return config, kwargs
    
    
def model_post_process(model, model_path, tokenizer, device="cuda"):
    model.eval()
    image_processor = None
    if is_mm_model(model_path):
        # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # if mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # if mm_use_im_start_end:
        #     tokenizer.add_tokens(
        #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        #     )
        # model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=model.device, dtype=torch.float16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=model.device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
    
    
    
def load_pretrained_model(
    model_args,
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    ####################################################################
    if is_mm_model(model_path):  # 加载llava模型
        # Load LLaVA model
        ## TODO @yunhao: mind fixing lora
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False, legacy=False
            )
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        ## TODO @yunhao: mind fixing this
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            cfg_pretrained = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            mm_config_wrapper(config, kwargs)
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, use_fast=False, legacy=False
                )
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
        else:
            config = AutoConfig.from_pretrained(model_path)
            # init config
            config.resume_path = model_path
            config.mm_vision_select_feature = model_args.mm_vision_select_feature
            config.mm_vision_select_layer = model_args.mm_vision_select_layer
            
            prepare_config_for_eval(config, kwargs)
            if "mpt" in model_name.lower():
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            elif "gemma" in model_name.lower():
                model = LlavaGemmaForCausalLM.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            else:
                # kentang-mit@: llama-2 model
                # config._attn_implementation = "flash_attention_2"
                model = DtosLmmForCausalLM( # 此处加载模型
                    config=config,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            tokenizer = model.tokenizer
    else:                        # 加载非llava模型
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False, legacy=False
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
    ####################################################################
    model.eval()
    image_processor = None
    if is_mm_model(model_path):
        # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # if mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # if mm_use_im_start_end:
        #     tokenizer.add_tokens(
        #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        #     )
        # model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device, dtype=torch.float16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def parse_model_name_or_path(config: PretrainedConfig, model_name="llm", suffix="_cfg"):
    target_model = f"{model_name}{suffix}"
    target_cfg = getattr(config, target_model, None)
    
    if isinstance(target_cfg, str):
        return target_cfg
    elif isinstance(target_cfg, dict):
        return target_cfg["architectures"][0]
    else:
        raise ValueError(f"Invalid {target_model} configuration!")

def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict): # 用于将vision_tower_cfg写入config
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")
    
    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    vision_tower_name = parse_model_name_or_path(config, "vision_tower")
    if "siglip" in vision_tower_name.lower():
        kwargs["device_map"] = "cuda"
        
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)