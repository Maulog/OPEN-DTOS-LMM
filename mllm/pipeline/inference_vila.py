'''
此文件暂时从vila中移植过来，todo 修改为适应所有模型的inference（参照nextchat的finetune.py）
'''

'''
Inference test to run all examples from the paper and compare w/ expected output.
Both the inference results and expected output will be printed out.

Currently do not support multi-turn chat. Each time an image and question are input and answer is output.
'''


import argparse
import os
import json
import re
import requests
from PIL import Image
from io import BytesIO

import torch
from PIL import Image
from mllm.config.constants import *
from mllm.config import prepare_args

from mllm.models.builder import load_pretrained
from mllm.dataset.utils import load_images

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# def image_parser(image_file, sep):
#     out = image_file.split(sep)
#     return out

def eval_model(cfg, model, tokenizer, image_processor):
    # Model
    disable_torch_init()
    
    image_files = cfg.data_args.image_files
    video_file = cfg.data_args.video_file

    if video_file is None:
        # image_files = image_parser(image_files, cfg.data_args.sep)
        images = load_images(image_files)
    else:
        if video_file.startswith("http") or video_file.startswith("https"):
            print("downloading video from url", video_file)
            response = requests.get(video_file)
            video_file = BytesIO(response.content)
        else:
            assert os.path.exists(video_file), "video file not found"
        from llava.mm_utils import opencv_extract_frames
        images = opencv_extract_frames(video_file, cfg.data_args.num_video_frames)
        
    model_name = get_model_name_from_path(cfg.model_args.model_name_or_path)

    qs = cfg.data_args.query # 提问，并将图片的token拼接到问题中
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    # qs = 'can you output json format?'
    print("input: ", qs)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if cfg.model_args.conv_args.conv_template is not None and conv_mode != cfg.model_args.conv_args.conv_template:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, cfg.model_args.conv_args.conv_template, cfg.model_args.conv_args.conv_template
            )
        )
    else:
        cfg.model_args.conv_args.conv_template = conv_mode

    conv = conv_templates[cfg.model_args.conv_args.conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

        
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(images_tensor.shape)
    
    gen_kwargs = cfg.data_args.gen_kwargs
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if gen_kwargs.temperature > 0 else False,
            temperature=gen_kwargs.temperature,
            top_p=gen_kwargs.top_p,
            num_beams=gen_kwargs.num_beams,
            max_new_tokens=gen_kwargs.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    
    image_processor = preprocessor['image']
    tokenizer = preprocessor['text']

    # tokenizer, model, image_processor, context_len = load_pretrained(args.model_name, None, "llava_llama") # 此处要修改，参考finetune.py
    result_list = eval_model(cfg, model, tokenizer, image_processor)


