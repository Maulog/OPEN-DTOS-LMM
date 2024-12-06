import copy
import random
import re
import sys
import logging
import typing
from typing import List, Dict, Any, Tuple, Union

import torch

from ..utils.transform import norm_rec, de_norm_rec

from ..root import (
    FUNCTIONS,
    BaseTargetProcessFunc,
)

from mllm.engine.registry import BOXES_PROCESSOR
from mllm.utils.span_utils import span_xx_to_cxw, span_cxw_to_xx
from ...utils import smart_tokenizer_and_partial_embedding_resize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]
random.seed(42)

@FUNCTIONS.register_module()
class RecFormatProcess(BaseTargetProcessFunc):
    def __call__(self, raw_item, mode, data_augmentation=False):
        clip_length = raw_item['clip'][1] - raw_item['clip'][0]
        raw_item['clip_length'] = clip_length
        raw_item['timestamps'] = raw_item['timestamps']
        raw_item['norm_timestamps'] = copy.deepcopy(raw_item['timestamps'])
        raw_item_copy = copy.deepcopy(raw_item)
        
        caption_num = len(raw_item['captions'])
        
        for i, times in enumerate(raw_item['timestamps']): # 第几个caption
            times = torch.tensor(times) # 当前caption对应的times
            norm_times = times / clip_length
            raw_item['timestamps'][i] = times
            raw_item['norm_timestamps'][i] = times / clip_length
            
            if data_augmentation and mode=='train':
                orig_times = times.clone()
                options = ['split', 'cat', 'none']
                probabilities = [0, 0, 1]
                aug_method = random.choices(options, weights=probabilities, k=1)[0]
                
                # if raw_item['source'] == 'activity_caption':
                #     probabilities = [0, 0.1, 0.9]
                #     aug_method = random.choices(options, weights=probabilities, k=1)[0]
                
                if aug_method == 'split' and len(times) == 1:
                    copy_num = random.randint(0, 2) # 复制数量  3
                    joint_num = min(2, count_split_strings(raw_item_copy['captions'][i])) # 是指切几刀 之前rand 0-2
                    # crop_num = random.randint(0, 3) # 裁剪数不超过复制数
                    
                    if copy_num != 0:
                        copy_times = orig_times.repeat(copy_num, 1)
                        times = torch.cat((times, copy_times))
                        
                    # 开始拼接（最后的并集是最大的预测）
                    if joint_num != 0 and (norm_times[..., 1]-norm_times[..., 0]) > 0.2: # 平均分，不裁太短的   0.3
                        split_points = torch.linspace(norm_times[0, 0], norm_times[0, 1], steps=joint_num+2)
                        joint_times = []
                        for j, _ in enumerate(split_points):
                            joint_times.append([split_points[j] ,split_points[j+1]])
                            if j+1 == len(split_points)-1:
                                break
                        joint_times = torch.tensor(joint_times)*clip_length
                        times = torch.cat((times, joint_times))
                        
                    '''
                    # # 开始中心裁剪
                    # if crop_num != 0 and (norm_times[..., 1]-norm_times[..., 0]) > 0.3: # 随机裁剪，不裁太短的
                    #     crop_scale = torch.tensor([random.uniform(0.3, 1) for _ in range(crop_num)])
                    #     crop_times = orig_times.repeat(crop_num, 1)
                        
                    #     norm_times_cxw = span_xx_to_cxw(crop_times)
                    #     norm_times_cxw[:, 1] *= crop_scale
                    #     crop_times = span_cxw_to_xx(norm_times_cxw)
                            
                    #     times = torch.cat((times, crop_times))
                    '''
                    
                    raw_item['timestamps'][i] = times
                    raw_item['norm_timestamps'][i] = times / clip_length
                    
                elif aug_method == 'cat' and caption_num > 1: # 一定会拼接
                    ## 将多个caption拼接为一个caption
                    cat_num = random.randint(1, min(1, caption_num-1)) # min 2
                    cat_candidate_idx_list = [j for j in range(caption_num) if i != j] # 候选
                    cat_idx_list = random.sample(cat_candidate_idx_list, cat_num)
                    cat_idx_list.append(i)
                    cat_idx_list = sorted(cat_idx_list) # 不需要排序，更混乱一些，提升鲁棒性
                    
                    times_list = []
                    for cat_idx in cat_idx_list:
                        if isinstance(raw_item_copy['timestamps'][cat_idx], torch.Tensor):
                            times_list.append(raw_item_copy['timestamps'][cat_idx])
                        else:
                            times_list.append(torch.tensor(raw_item_copy['timestamps'][cat_idx]))
                    times = torch.cat(times_list) # [n,2]
                    cat_caption = ', '.join([raw_item_copy['captions'][cat_idx] for cat_idx in cat_idx_list])
                    
                    raw_item['timestamps'][i] = times # 这里返回应该是[n,2]的tensor
                    raw_item['norm_timestamps'][i] = times / clip_length
                    raw_item['captions'][i] = cat_caption
                    pass
                
                elif aug_method == 'none':
                    copy_num = random.randint(0, 3) # 复制数量 
                    if copy_num != 0:
                        copy_times = orig_times.repeat(copy_num, 1) 
                        times = torch.cat((times, copy_times))
                    raw_item['timestamps'][i] = times
                    raw_item['norm_timestamps'][i] = times / clip_length
                
                
        
        return raw_item['timestamps'], raw_item['norm_timestamps'], \
               raw_item['clip_length'], raw_item['captions']

def count_split_strings(text):
    substrings = ["and", "then", "while"]
    words = re.split(r'[,\s]+', text) # ,
    total_count = 0
    for word in words:
        if word in substrings:
            total_count += 1

    comma_count = text.count(',')

    return total_count + comma_count







