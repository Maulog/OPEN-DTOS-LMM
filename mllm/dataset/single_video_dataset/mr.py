import json
import random
import os
import re
import sys
from typing import Dict, Any, Callable, List, Optional, Tuple, Type, Sequence
import logging
import warnings
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler

from mllm.dataset.utils.compute_metrics import BaseComputeMetrics
from mllm.dataset.utils.image_grid import ImageGrid
from mllm.utils.box_ops import rec_iou_in_caption

from ..root import (
    DATASETS,
    METRICS,
)
from mllm.dataset.utils.io import read_jsonl_file, read_json_file, load_video, read_txt_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

image_grid = ImageGrid()

class CompleteSampler(Sampler): # trainer中用不了
    def __init__(self, data_source, max_k=4):
        self.data_source = data_source
        self.max_k = max_k
        self.used_indices = set()  # 用于跟踪已经完全使用的数据点索引

    def __iter__(self):
        data_indices = list(range(len(self.data_source)))
        random.shuffle(data_indices)  # 随机化索引顺序
        for idx in data_indices:
            if idx not in self.used_indices:
                captions_indices = list(range(len(self.data_source[idx]['captions'])))
                random.shuffle(captions_indices)  # 随机化索引顺序
                while len(captions_indices) > 0:
                    k = min(random.randint(1, self.max_k), len(captions_indices))
                    yield idx, captions_indices[:k]
                    captions_indices = captions_indices[k:]
                self.used_indices.add(idx)  # 标记当前数据点已完全使用
                
def pre_process_captions(captions: List[str]) -> List[str]: # 去除结束符号的标点
    punctuation_pattern = re.compile(r'[^\w\s]$')
    for i in range(len(captions)):
        captions[i] = captions[i].replace('\n', '').replace('\t', '').strip() # 去除换行符和制表符和收尾空白符
        captions[i] = punctuation_pattern.sub('', captions[i]) # 去除末尾标点
        captions[i] = ' '.join(captions[i].split()) # 去除多余空格
        captions[i] = captions[i].lower()
    return captions
                
                
@DATASETS.register_module()
class MomentRetrievalTreeDataset(Dataset): 
    # TODO: 添加提问模板，继承混合模板的数据集类
    _repr_indent = 4
    def __init__(self, filename, template_file, video_folder, feat_folder, use_video, n_frms=None, *args, **kwargs): 
        # super().__init__(*args, **kwargs)
        self.filename = filename
        self.video_folder = video_folder
        self.feat_folder = feat_folder
        self.n_frms = n_frms
        self.is_tree = True if 'tree' in self.filename else False
        self.is_first = True
        self.mock = None
        self.frame_num_folder = 'ig100_im10_sub9' # 'ig100_im10_sub9'
            
        self.use_video = use_video
        
        self.data = read_jsonl_file(self.filename)
        if os.path.exists(os.path.join(self.feat_folder, self.frame_num_folder, 'invalid_videos.txt')):
            invalid_video_path = os.path.join(self.feat_folder, self.frame_num_folder, 'invalid_videos.txt')
            invalid_video_names = set(read_txt_file(invalid_video_path, return_str=False))
            self.data = [i for i in self.data if i['video'].rsplit('.', 1)[0] not in invalid_video_names]
            
        self.prefix = read_txt_file(template_file, return_str=True)
        
    def get_video(self, video_path) -> List[Image.Image]: # 这里应该添加如果读取失败的逻辑
        
        frms_list, indices, fps, vlen = load_video(video_path, n_frms=100, interpolated_frames=True) # self.n_frms
        # if frms_list==[]:
        #     raise ValueError(f"Load video failed: {video_path}")
        return frms_list
        
    def get_video_feat(self, video_feat_path) -> torch.Tensor: # 注意dtype
        video_feat_np = np.load(video_feat_path)
        video_feat = torch.from_numpy(video_feat_np)
        return video_feat
    
    def get_image_feat(self, image_feat_path) -> torch.Tensor: # 注意dtype
        image_feat_np = np.load(image_feat_path)
        image_feat = torch.from_numpy(image_feat_np)
        return image_feat
        
        
    def __getitem__(self, data_index, debug_mode=False, return_conv=False) -> Dict[str, Any]: 
        
        item = self.data[data_index]
        
        # 这里不能直接用item，因为item是引用，会改变原数据，相当于在原数据上不断添加数据而不进行删除
        ret_dict = item.copy() # 这里一定要加，控制内存管理
        
        if not self.is_tree:
            ret_dict['captions'] = [ret_dict['captions']]
            ret_dict['timestamps'] = [ret_dict['timestamps']]
        
        ret_dict['captions'] = pre_process_captions(ret_dict['captions']) # 不泄露
        
        
        video_feat = None
        image_feat = None
        img_frms = None
        vid_frms = None
        frms_list = None
        

        if self.use_video: # 这里记得改
            video_path = os.path.join(self.video_folder, ret_dict['video'])
            frms_list = self.get_video(video_path) # 总的提取帧数
            
            vid_ids, img_ids = image_grid.get_video_and_image_ig_id(np.arange(0, 100, 1), image_num=10, rough_img_num_sub_interval=9)
            
            img_frms = [frms_list[img_id] for img_id in img_ids]
            vid_frms = []
            for vid_id_interval in vid_ids:
                tmp_vid_frms = []
                for vid_id in vid_id_interval:
                    tmp_vid_frms.append(frms_list[vid_id])
                vid_frm = image_grid.get_single_image_grid(tmp_vid_frms, image_rows=3, image_cols=3)
                vid_frms.append(vid_frm)
            
            if frms_list == []: # 如果出现了空值则用mock数据填充，只要数据中少部分是坏值就用这个方法代替
                print('Load video failed:', video_path)
                with open('invalid_videos.txt', 'a') as f:
                    f.write(video_path + '\n')
                return self.mock
            # video = [frms_list[i] for i in [9, 29, 49, 69, 89]]
        else:
            video_feat_path = os.path.join(self.feat_folder, self.frame_num_folder, 'video', ret_dict['video'].rsplit('.', 1)[0] + ".npy")
            image_feat_path = os.path.join(self.feat_folder, self.frame_num_folder, 'image', ret_dict['video'].rsplit('.', 1)[0] + ".npy")
            try:
                video_feat = self.get_video_feat(video_feat_path)
                image_feat = self.get_image_feat(image_feat_path) 
            except:
                print('Load feature failed:', video_feat_path)
                with open('invalid_videos.txt', 'a') as f:
                    f.write(video_feat_path + '\n')
                return self.mock

                       
            
        ret_dict['prefix'] = self.prefix # 不泄露内存
        
        ret_dict['vi_src'] = vid_frms
        ret_dict['im_src'] = img_frms   # 不泄露内存
        
        
        ret_dict['vi_feat'] = video_feat
        ret_dict['im_feat'] = image_feat # 二选一为None
        
        # torch.save(ret_dict, 'mock.pth')
        if self.is_first: # 将每个数据集的第一个数据保存下来，如果遇到视频读取失败的情况就用self.mock作为填充
            self.mock = ret_dict.copy()
            self.is_first = False
            
        return ret_dict
        

    
    def __len__(self):
        return len(self.data)
    
    
@METRICS.register_module()
class MRComputeMetrics(BaseComputeMetrics): # 应该参考pope的
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def __call__(self, preds_and_targets) -> Dict[str, Any]:
        return self.calculate_metric(preds_and_targets)
    def calculate_metric(self, preds_and_targets: list[torch.Tensor]):
        preds, targets = preds_and_targets
        
        
        pred_recs_and_scores = preds['pred_ts']
        pred_recs, pred_scores = pred_recs_and_scores[..., :2], pred_recs_and_scores[..., -1]
        
        target_recs = targets['label_ts']
        
        extract_failed = 0

        with torch.no_grad(): # 这里注意数据格式
            ious = []
            for i, (target_rec, pred_rec) in enumerate(zip(target_recs, pred_recs)):
                if preds['pred_num'][i] == 0: # 一般不太会发生
                    extract_failed += 1
                    continue
                
                pred_rec = pred_rec[:preds['pred_num'][i]]
                target_rec = target_rec[:targets['label_num'][i]]
                
                iou = rec_iou_in_caption(torch.tensor(pred_rec), torch.tensor(target_rec)) # 此处原来有 * 1000，参考rec
                iou = iou if type(iou) == torch.Tensor else torch.tensor(iou)
                ious.append(iou) 
                
            ious = torch.stack(ious)
            
        # NOTE: please note iou only calculate for success target
        return {
            'R@0.3': 1.0 * (ious > 0.3).sum().item() / (len(target_recs)-extract_failed),
            'R@0.5': 1.0 * (ious > 0.5).sum().item() / (len(target_recs)-extract_failed),
            'R@0.7': 1.0 * (ious > 0.7).sum().item() / (len(target_recs)-extract_failed),
            'miou': ious.mean().item(),
            'extract_failed': extract_failed,
        }



    # 将提取的代码放在generate中
    # def extract_ans(self, string: str, target_captions: list[str]) -> Optional[Dict[str, Any]]: # 这里没检查
    #     # TODO: 更改提取的策略，转换为字典可能会出错，导致提取出错较多
    #     try:
    #         pattern = r'\{.*?\}'
    #         match = re.search(pattern, string)
    #         if match is None:
    #             raise ValueError("No target dict found in the output.")
    #         target_dict_str = match.group(0) # 如果是none也会引发报错
    #         try:
    #             target_dict = json.loads(target_dict_str)
    #         except json.JSONDecodeError as e:
    #             raise ValueError("Target dict is not a valid json string.")
                
    #         ret = {}
    #         for k, v in target_dict.items():
    #             for caption in target_captions:
    #                 if k in caption:
    #                     ret[caption] = v
    #         return ret
    #     except Exception as e:
    #         logger.warning(f"{e}")
    #         return None

    
    
# class DataRegister: # 用于存放从数据中取到但是没用完的数据，同时可以限制最大采样数量
#     def __init__(self, max_sample_num):
#         self.max_sample_num = max_sample_num
#         self.item = None
        
#     def set_item(self, item):
#         if self.is_empty():
#             self.item = copy.deepcopy(item)
#         else:
#             raise ValueError('DataRegister is not empty')
        
#     def pick_caption_and_timestamp(self):
#         if self.is_empty():
#             raise ValueError("DataRegister is empty, can't pick caption.")
#         existing_num = len(self.item['captions'])
#         cur_sample_num = random.randint(1, max(self.max_sample_num, existing_num))
#         captions_list = [self.item['captions'].pop() for _ in range(cur_sample_num)]
#         timestamps_list = [self.item['timestamps'].pop() for _ in range(cur_sample_num)]
        
#         ret = {
#             'video': self.item['video'],
#             'duration': self.item['duration'],
#             'source': self.item['source'],
#             'clip': self.item['clip'],
#             'captions': captions_list,
#             'timestamps': timestamps_list,
#         } 
#         return ret
        
#     def is_empty(self):
#         if self.item is None or self.item['captions'] == []:
#             return True 
#         return False    