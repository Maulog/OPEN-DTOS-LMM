import sys
import copy
import warnings
import logging
from typing import Dict, Any, List

import PIL.Image
import torch
from PIL import Image
from transformers import LlamaTokenizer

from llava.mm_utils import process_images

from ..root import (
    FUNCTIONS,
    # IMAGE_PLACEHOLDER,
    BaseImageProcessFunc,
    BaseConvProcessFunc,
    BaseTextProcessFunc,
)
from mllm.config.constants import *
from llava.conversation import Conversation, SeparatorStyle


# IGNORE_INDEX = -100
# DEFAULT_IMAGE_TOKEN = IMAGE_PLACEHOLDER
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
# DEFAULT_AT_TOKEN = "<at>"
# DEFAULT_BOXES_TOKEN = "<boxes>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@FUNCTIONS.register_module()
class ChatConvProcess(BaseConvProcessFunc):
    def __call__(self, raw_conv: List[Dict[str, Any]], preprocessor: Dict[str, Any], conv_template: Conversation) -> List[Dict[str, Any]]:
        conv_processor_cfg = preprocessor['conv']

        image_token_len = conv_processor_cfg['image_token_len']
        video_token_len = conv_processor_cfg['video_token_len']
        sep_image_conv_front = conv_processor_cfg.get('sep_image_conv_front', False) # 把图像放到对话的最前面
        use_im_start_end = conv_processor_cfg.get('use_im_start_end', False)


        # if sep_image_conv_front: # 有问题没改，暂不用
        #     raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        #     raw_conv[0]['value'] = DEFAULT_IMAGE_TOKEN + conv_template.sep + conv_template.roles[0] + ": " + raw_conv[0]['value']
        if use_im_start_end:
            for message in raw_conv.messages:
                # sentence = message[1]
                if message[1] is None:
                    continue
                    
                # 注意此处不要加空格，空格也是token
                replace_video_sentence = DEFAULT_VI_START_TOKEN + VIDEO_PLACEHOLDER*video_token_len + DEFAULT_VI_END_TOKEN + ' \n'
                replace_frame_sentence = DEFAULT_IM_START_TOKEN + IMAGE_PLACEHOLDER*image_token_len + DEFAULT_IM_END_TOKEN + ' \n'
            
                message[1] = message[1].replace(DEFAULT_VIDEO_TOKEN, replace_video_sentence)
                message[1] = message[1].replace(DEFAULT_IMAGE_TOKEN, replace_frame_sentence)

        return raw_conv


@FUNCTIONS.register_module()
class ChatTextProcess(BaseTextProcessFunc): # 用于读取conv类中的信息，进行相关的截断、掩码、label提取等

    def __call__(self, conv: Conversation, preprocessor: Dict[str, Any], mode: str, **tokenize_kwargs) -> Dict[str, Any]:
        tokenizer = preprocessor['text']
        # assert isinstance(tokenizer, LlamaTokenizer), "only work for LlamaTokenizer"

        _truncation_size = tokenize_kwargs.pop('truncation_size', None)
        _kwargs = {'return_tensors': 'pt'}
        _kwargs.update(tokenize_kwargs)

        if conv.sep_style == SeparatorStyle.LLAMA_3:
            if mode in ['train']:
                ret = self.tk_conv_colon_two_train(conv, tokenizer, **_kwargs) # 这个处理多轮对话的函数要重写
            else:
                ret = self.tk_conv_colon_two_eval(conv, tokenizer, **_kwargs) # 用于预测基于上下文的最后一条输出
        else:
            raise ValueError(f"unrecognized conv_style: {conv.sep_style}.\n the conv is {conv}")

        if _truncation_size is None:
            return ret
        if len(ret['input_ids']) <= _truncation_size: # 截断长度默认4096
            return ret

        # 开始截断
        origin_len = len(ret['input_ids'])
        ids_to_remove_num = origin_len - _truncation_size  # 截断（截断不重要的文本，保留图像相关的ids）
        # truncation. should carefully not truncate <img_token>
        ids_should_not_remove = list(map( # 图像相关的，target相关的token ids
            tokenizer.convert_tokens_to_ids,
            (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_MOMENT_TOKEN,
             IMAGE_PLACEHOLDER, DEFAULT_VIDEO_TOKEN, VIDEO_PLACEHOLDER, DEFAULT_VI_START_TOKEN, 
             DEFAULT_VI_END_TOKEN,)
        )) 
        back_no_image = all(ids not in ids_should_not_remove for ids in ret['input_ids'][_truncation_size:]) # bool 如果所有ids不包含图像相关的ids，返回true
        if back_no_image: # 没有图像数据
            tgt_ids = list(range(_truncation_size)) # 从0开始生成截断长度的序号
        else:
            ids_to_remove = set() # 除与图像target相关的关键ids中的idx（不关键）
            # for idx in range(origin_len - 1, -1, -1): # 倒序，从末尾截断
            for idx in range(45, origin_len): # 不截系统提示，和<user>的开头
                if ret['input_ids'][idx] not in ids_should_not_remove: # 不在与图像target相关的关键ids中
                    ids_to_remove.add(idx) # 添加idx
                    if len(ids_to_remove) >= ids_to_remove_num:
                        break
            tgt_ids = [_ for _ in range(origin_len) if _ not in ids_to_remove] # 图像的idx进行保存
        logger.warning(f"truncate sample size from {origin_len} to {len(tgt_ids)}.")
        assert len(tgt_ids) == _truncation_size, f"{len(tgt_ids)}, {_truncation_size}, {ret['input_ids'].tolist()}"
        truncated_ret = {k: v[tgt_ids] for k, v in ret.items()} # 把ret所有的属性都返回，筛选包含图像的input_ids
        return truncated_ret

    # noinspection PyMethodMayBeStatic
    def tk_conv_colon_two_train(self, conv, tokenizer, **kwargs): # 将输入token化
        conversation = conv.get_prompt() # 完整的文本输入
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0] # tokenize
        target = copy.deepcopy(input_ids)
        assert conv.sep_style == SeparatorStyle.LLAMA_3
        # Mask targets
        # sep = conv.sep + conv.roles[1] + ": "
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # 求不等于pad的长度
        rounds = conversation.split(conv.sep) # 将多轮对话分割为单轮对话
        cur_len = 1 # <|begin_of_text|>
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds): # 将system的输出变为IGNORE_INDEX
            if rou == "":
                break
            if i % 2 == 0 and i !=0 : # 系统回答
                round_len = len(tokenizer(rou).input_ids) + 1 # <|end_of_text|>
                target[cur_len: cur_len + 4] = IGNORE_INDEX # 只把前缀mask,<|start_header_id|>user<|end_header_id|>\n\n
            elif i==0: # 第一句是系统提示
                round_len = len(tokenizer(rou).input_ids) # 此处不+1是因为第一句中已经有<|begin_of_text|>
                target[cur_len: cur_len + round_len] = IGNORE_INDEX
            else: # 用户提示
                round_len = len(tokenizer(rou).input_ids) + 1 # <|end_of_text|>
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
            cur_len += round_len     
        # target[cur_len-1:] = IGNORE_INDEX # -1是因为最后一轮没有<|end_of_text|>不会被分割
        # 如果有结束符将其mask
        
        
        
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            labels=target,
        )

    # noinspection PyMethodMayBeStatic
    def tk_conv_colon_two_eval(self, conv, tokenizer, **kwargs): # 仅处理单轮对话
        assert len(conv.messages) >= 2
        # target = conv.messages[-1][-1]
        target = conv.get_prompt()

        conv.messages[-1][-1] = "" # 清除最后一条信息作为输入
        conversation = conv.get_prompt()
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0] # 不包含label的信息

        target = tokenizer([target, ], add_special_tokens=False, **kwargs).input_ids[0] # 全对话信息
        target[target == tokenizer.pad_token_id] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id), # 返回不等于pad_token_id为1，其余为0的掩码
            labels=target,
        )


@FUNCTIONS.register_module()
class ChatImageProcessor(BaseImageProcessFunc):
    def __call__(self, images: Image.Image, preprocessor: Dict[str, Any], model_cfg:Dict[str, Any]) -> Dict[str, Any]:
        image_processor = preprocessor['image']

        image_tensor =  process_images(images, image_processor, model_cfg)
        
        return image_tensor
    
    
        # if isinstance(image, (list, tuple)):
        #     image = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        #     assert False, 'dtos not support MultiImage'
        # elif isinstance(image, PIL.Image.Image):
        #     image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # else:
        #     if hasattr(image_processor, 'crop_size'):
        #         crop_size = image_processor.crop_size
        #         height, width = crop_size['height'], crop_size['width']
        #     else:
        #         raise ValueError("got empty image. and don't know how to pad")
        #     image = torch.zeros(3, height, width)
        # return {'image': image}
