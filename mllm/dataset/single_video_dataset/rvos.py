###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
import copy
from pathlib import Path
import re
from typing import Dict, Any, Callable, List, Optional, Tuple, Type, Sequence

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
# import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random
import sys

from mllm.dataset.single_video_dataset.mevis_metrics import db_eval_iou, db_eval_boundary
from mllm.dataset.single_video_dataset.mr import pre_process_captions
from mllm.dataset.utils.compute_metrics import BaseComputeMetrics
from mllm.dataset.utils.transform import norm_box_xyxy
from mllm.dataset.utils.refytvos_utils import load_mevis_json, load_refytvos_json, load_refdavis_json
from mllm.models.sam2.sav_benchmark import Evaluator



from ..root import (
    DATASETS,
    METRICS,
)
from mllm.dataset.utils.image_grid import ImageGrid
from mllm.dataset.utils.io import read_txt_file
sys.path.append('/home/tianjirui/DTOS-LMM')


from pycocotools import mask as coco_mask

from mllm.utils.box_ops import merge_overlapping_intervals

image_grid = ImageGrid()

random.seed(42)

def prepare_metas(base_folder, mode, dataset_name): # 用于创建metas，和self.video
    if dataset_name == 'mevis':
        metas, mask_dict = load_mevis_json(base_folder, mode)
    elif dataset_name == 'youtubervos':
        ann_file = os.path.join(base_folder, 'meta_expressions.json')
        metas, mask_dict, _ = load_refytvos_json(base_folder, ann_file, dataset_name, mode)
    elif dataset_name == 'davis':
        ann_file = os.path.join(base_folder, 'meta_expressions.json')
        metas, mask_dict, _ = load_refdavis_json(base_folder, ann_file, dataset_name, mode)
    
    
    videos = set([meta['video'] for meta in metas])
    
    return metas, videos, mask_dict

def get_dataset_mode(dataset_name, base_folder):
    if dataset_name == 'mevis':
        if 'valid_u' in str(base_folder):
            mode = 'val'
        elif 'train' in str(base_folder):
            mode = 'train'
        elif 'example' in str(base_folder):
            mode = 'example'
        else:
            mode = 'test'
    elif dataset_name == 'davis':
        if 'valid' in str(base_folder):
            mode = 'val'
        elif 'train' in str(base_folder):
            mode = 'train'
    elif dataset_name == 'youtubervos':
        if 'valid' in str(base_folder):
            mode = 'val'
        elif 'train' in str(base_folder):
            mode = 'train'
        elif 'test' in str(base_folder):
            mode = 'test'
    return mode


def format_moment_preds(moment_preds, metas, dynamic_path): 
    # 重新整理，提取有效信息，设置策略(这里有点慢，看下怎么改并且出现了越界的问题)
    # 该函数是从stage1中的结果中重组信息，变为需要的信息
    ret = {}
    not_found_list = []
    
    duplicate_keys = set() # 这里经过预处理之后会有重复的key（可能是大小写不同，也可能是首尾空格不同）
    # 构建 mp_map
    mp_map = {}
    for mp in moment_preds:
        key = f"{mp['video']}_{pre_process_captions(mp['captions'])[0]}"
        if key in mp_map:
            duplicate_keys.add(key)
        else:
            mp_map[key] = mp
    
    for meta in metas: # 遍历目标数据更容易找到缺失数据
        key = f"{meta['video']}_{pre_process_captions([meta['exp']])[0]}"
        if key in mp_map:
            mp = mp_map[key]
            st_ed, score = mp['pred_ts'][0][:mp['pred_num']][:, :2], mp['pred_ts'][0][:mp['pred_num']][:, -1]
            if mp['video'] not in ret:
                ret[mp['video']] = {}
            ret[mp['video']][meta['exp_id']] = st_ed # 均是新创建属性
        else:
            not_found_list.append(meta)
                    
            
    if len(not_found_list) > 0:
        print('\n********some data not correctly matched, please adjust match strategy********\n')
        torch.save(not_found_list, os.path.join(dynamic_path.rsplit('/',1)[0], 'not_matched_data.pth'))  
        print('not_matched_data saved to ', os.path.join(dynamic_path.rsplit('/',1)[0], 'not_matched_data.pth'))
        print('*****************train model used matched data only*****************************\n')  
        
    assert sum([len(mp_value) for mp_value in ret.values()]) == len(metas), 'all moment predictions should be matched'
    
    return ret

def replicate_items_to_length(items, target_length):
    num_items = len(items)
    new_items = list(items)
    
    remaining_length = target_length - num_items
    
    while len(new_items) < target_length:
        item = random.choice(items)
        new_items.append(item)
    
    return new_items


@DATASETS.register_module()
class RVOSDataset(Dataset): 
    # 这里ann_ids是用来在mask_dict.json文件中查询的，而obj_id是在这个视频中对应的实例的id
    # 因此ann_ids和obj_id是一一对应的，数量是相等的，且ann_ids是在递增的（有时候不变，不增）
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, base_folder: Path, n_frms: int, sampling_type: str = 'Dynamic',
                 dynamic_path: str = None, template_file: str = None, use_video: bool = True,
                 feat_folder: str = None, *args, **kwargs):
        self.base_folder = base_folder
        self.ann_file = os.path.join(self.base_folder, 'meta_expressions.json')
        self.num_frames = n_frms # 指要从一个视频中取出几帧(也是最优采样帧)
        self.use_video = use_video
        self.feat_folder = feat_folder
        self.mode = None
        self.mock = None
        self.is_first = True
        dataset_name_list = ['mevis', 'davis', 'youtubervos']
        for dataset_name in dataset_name_list:
            if dataset_name in str(self.base_folder).lower(): # 这里要注意大小写还有缩写
                self.dataset_name = dataset_name
                break
        
        self.vid_target_len = 300 # 视频帧插帧到固定帧数
        
        # sampling type
        assert sampling_type in ['Dynamic', 'Uniform', 'Random', 'All', 'Dynamic_and_Uniform'], \
            f'sampling type should be Dynamic, Uniform, Random or All but got {sampling_type}'
        self.sampling_type = sampling_type
        
        self.prefix = read_txt_file(template_file, return_str=True)
        
        
        ###### 这里下面包装成另一个函数 ######
        self.mode = get_dataset_mode(self.dataset_name, base_folder)

        if dynamic_path != None:
            if self.mode == 'train':
                self.dynamic_path = os.path.join(dynamic_path, f'multitest_{self.dataset_name}_prediction.pth')
            else:
                self.dynamic_path = os.path.join(dynamic_path, f'multitest_{self.dataset_name}_{self.mode}_prediction.pth')

        # create video meta data
        self.metas, self.videos, self.mask_dict = prepare_metas(self.base_folder, self.mode, self.dataset_name)
        
        if self.sampling_type == 'Dynamic' or self.sampling_type == 'Dynamic_and_Uniform': 
            assert self.dynamic_path is not None, 'dynamic path should be provided for dynamic sampling'
            raw_moment_preds = torch.load(self.dynamic_path, map_location=torch.device('cpu')) # 初始化不能在cuda
            self.moment_preds = format_moment_preds(raw_moment_preds, self.metas, self.dynamic_path)
        
        
        
        # show in terminal
        print('<<<')
        print('dataset_name: ', self.dataset_name, 'video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('>>>')
            

    def get_image_feat(self, image_feat_path) -> torch.Tensor: # 注意dtype
        image_feat_np = np.load(image_feat_path)
        image_feat = torch.from_numpy(image_feat_np)
        return image_feat

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # instance_check = False
        # while not instance_check:
        meta = self.metas[idx]  # dict
        meta_copy = meta.copy() # 控制内存

        video, exp, exp_id, anno_id, category, frames = \
            meta_copy['video'], meta_copy['exp'], meta_copy['exp_id'], meta_copy['anno_id'], meta_copy['category'], meta_copy['frames']
        # clean up the caption
        exp = pre_process_captions([exp])[0] # 变为小写，用空格分开
        category_id = 0
        vid_len = len(frames)
        
        
        # 检查长度，将比较短的视频帧插帧到比较长的长度，之后再进行采样
        # vid_index为插帧后的索引
        if vid_len < self.vid_target_len:
            vid_index = np.linspace(0, vid_len - 1, self.vid_target_len).astype(int)
            # target_fixed_frames = [frames[i] for i in target_vid_index]
        else: # 超过比较长的直接采样
            vid_index = np.arange(0, vid_len)
        interpreted_length = len(vid_index)
        assert interpreted_length >= vid_len, f'interpreted_length must be more than vid_len'
                
        
        # different sample methods, only compute index
        sample_vid_indx = None
        per_moment_sample_num = None
        if self.sampling_type == 'Dynamic':
            # try: # 后续计划取消这里的try except，应该在初始化的时候检查完毕
            sample_indx, per_moment_sample_num = self._st_ed_map_to_frames(self.moment_preds[video][exp_id], vid_index) # 用插帧后的索引作为输入
            # except:
            #     print(f'video: {video}, exp_id: {exp_id} not found in moment_preds')
            #     with open('invalid_moment_preds_stage2.txt', 'a') as f:  
            #         f.write(f'video: {video}, exp_id: {exp_id} \n')
            #     return self.mock
        elif self.sampling_type == 'Uniform': # 只有这里会用到stage1推理的视频模态
            # sample_indx = np.linspace(0, interpreted_length - 1, self.num_frames, dtype=int)
            # sample_indx = [vid_index[idx] for idx in sample_indx]
            indices = np.arange(0, interpreted_length)
            sample_vid_indx, sample_indx = image_grid.get_video_and_image_ig_id(indices, image_num=self.num_frames, rough_img_num_sub_interval=9)
            sample_indx = [vid_index[idx] for idx in sample_indx]
            sample_vid_indx = [vid_index[idx] for idx in sample_vid_indx]
        elif self.sampling_type == 'Random': # TODO: 可能要添加sample_vid_indx
            # 这里或者用 vid_index （重复多） 或者用np.arange(vid_len)（重复少，但可能有问题，如果vid_len小于self.num_frames可能有问题）
            sample_indx = np.random.choice(vid_index, size=self.num_frames, replace=True) 
        elif self.sampling_type == 'All': # 不计划使用（会超过长度
            sample_indx = np.arange(0, vid_len)
        elif self.sampling_type == 'Dynamic_and_Uniform':
            cur_sample_type = random.choice(['Dynamic', 'Uniform'])
            if cur_sample_type == 'Dynamic':
                sample_indx, per_moment_sample_num = self._st_ed_map_to_frames(self.moment_preds[video][exp_id], vid_index)
            elif cur_sample_type == 'Uniform':
                indices = np.arange(0, interpreted_length)
                sample_vid_indx, sample_indx = image_grid.get_video_and_image_ig_id(indices, image_num=self.num_frames, rough_img_num_sub_interval=9)
                sample_indx = [vid_index[idx] for idx in sample_indx]
                sample_vid_indx = [vid_index[idx] for idx in sample_vid_indx]
        else:
            raise NotImplementedError(f'Unknown sampling type {self.sampling_type}')

        sample_indx = np.sort(np.array(sample_indx)) # 这里是取图片的索引（random、uniform、dynamic）
        


        # read frames and masks
        imgs, vids, labels, boxes, masks, all_masks, valid, image_feats = [], [], [], [], [], [], [], []
        
        if not self.use_video: 
            all_image_feats = self.get_image_feat(self.feat_folder, 'pool_n25', video) # 这里待明确
            
        if sample_vid_indx is not None: # 处理vids
            for vid_id_interval in sample_vid_indx:
                tmp_vid_frms = []
                for vid_id in vid_id_interval: # 这里需要提取其中每一帧图片的特征
                    frame_name = frames[vid_id]
                    img_path = os.path.join(str(self.base_folder), 'JPEGImages', video, frame_name + '.jpg')
                    img = Image.open(img_path).convert('RGB')
                    tmp_vid_frms.append(img)
                vid_frm = image_grid.get_single_image_grid(tmp_vid_frms, image_rows=3, image_cols=3)
                vids.append(vid_frm)
            
        for j in range(len(sample_indx)): # 处理采样的帧
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            
            if self.use_video:
                img_path = os.path.join(str(self.base_folder), 'JPEGImages', video, frame_name + '.jpg')
                # mask_path = os.path.join(str(self.base_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                # h, w = img.shape
                image_feat = None
            else:
                image_feat = all_image_feats[frame_indx]
                img = None
            
            # 构造整个画面的mask
            mask = np.zeros(img.size[::-1], dtype=np.float32)
            if self.mask_dict is not None:
                for x in anno_id: # 此处将每一个ann叠加到一起
                    frm_anno = self.mask_dict[x][frame_indx] # self.mask_dict[x]是个长度为总帧长度的列表
                    if frm_anno is not None:
                        mask += coco_mask.decode(frm_anno)

            # create the target
            label = torch.tensor(category_id)

            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:  # some frame didn't contain the instance
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)
            mask = torch.from_numpy(mask)

            # append
            imgs.append(img)
            image_feats.append(image_feat)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)

        # 构造all_masks,用于自己进行评估
        for i in range(vid_len):
            frame_name = frames[i]
            mask = np.zeros(img.size[::-1], dtype=np.float32)
            if self.mask_dict is not None:
                for x in anno_id:
                    frm_anno = self.mask_dict[x][i]
                    if frm_anno is not None: # frm_anno是字典，有size和counts两个键
                        mask += coco_mask.decode(frm_anno)
            all_masks.append(torch.from_numpy(mask))
        all_masks = torch.stack(all_masks, dim=0) # .to(dtype=torch.bool).numpy()
        
        # 构造一个字典，用于单独保存每一个obj的mask、box、valid信息
        objs_masks_dict = {} 
        for i, obj_id in enumerate(anno_id):
            obj_mask = []
            obj_boxes = []
            obj_norm_boxes = []
            obj_valid = []
            objs_masks_dict[obj_id] = {}
            if self.mask_dict is not None:
                for j in sample_indx: # zh
                    frm_anno = self.mask_dict[obj_id][j]
                    if frm_anno is not None:
                        mask = coco_mask.decode(frm_anno)
                        y1, y2, x1, x2 = self.bounding_box(mask)
                        obj_box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                        obj_valid.append(1)
                    else:
                        mask = np.zeros(img.size[::-1], dtype=np.float32)
                        obj_box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                        obj_valid.append(0)
                    obj_boxes.append(obj_box)
                    obj_mask.append(torch.from_numpy(mask))
                objs_masks_dict[obj_id]['label_mask'] = torch.stack(obj_mask, dim=0)
                objs_masks_dict[obj_id]['label_boxes'] = torch.stack(obj_boxes, dim=0)
                objs_masks_dict[obj_id]['label_norm_boxes'] = norm_box_xyxy(torch.stack(obj_boxes, dim=0), w=img.size[0], h=img.size[1])
                objs_masks_dict[obj_id]['label_valid'] = obj_valid
        
        # 构造 tgt_norm_bbox_list
        tgt_norm_bbox_list = []
        # tgt_norm_bbox 的提取，这里还应该有判断当前帧是否有对应的目标的逻辑
        if self.mask_dict is not None:
            for i, tgt_id in enumerate(sample_indx): # 每一帧
                tgt_norm_bbox = [] # 包含多个目标检测框的列表
                for obj in objs_masks_dict.values(): # 每个目标
                    # 先检查对应的id是否可行
                    if obj['label_valid'][i]:
                        one_obj_tgt_norm_box = obj['label_norm_boxes'][i]
                        tgt_norm_bbox.append(one_obj_tgt_norm_box)
                    
                # 数据增强
                if len(tgt_norm_bbox) > 0 and len(tgt_norm_bbox) <= 5 and self.mode == 'train': # 如果有目标，且目标数小于5，则进行复制扩充数据
                    # 先随机最大数
                    max_copy_num = random.randint(len(tgt_norm_bbox), 6) # 这里是需要扩充到的数量
                    tgt_norm_bbox = replicate_items_to_length(tgt_norm_bbox, max_copy_num)
                
                tgt_norm_bbox_list.append(tgt_norm_bbox)

        # transform
        w, h = img.size
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        target = {
            'prefix': self.prefix,
            'im_src': None if all(img is None for img in imgs) else imgs,
            'vi_src': None if all(vid is None for vid in vids) else vids,
            'im_feat': None if all(image_feat is None for image_feat in image_feats) else image_feats,
            'vi_feat': None, # 待添加
            'captions': [exp],
            'exp_id': exp_id,
            'dataset_type': self.mode,
            'timestamps': [[[0, 150]]], # (stage1 label) dummy data, use tree logic
            'clip': [0, vid_len],
            'source': self.dataset_name,
            'video': video,
            # 此处要返回clip，用于计算clip_length！！！
            
            'video_dir': os.path.join(str(self.base_folder), 'JPEGImages', video),
            'frames_idx': torch.tensor(sample_indx),  # [T,] # 应该乱序这个采样的idx!!!
            'per_moment_sample_num': per_moment_sample_num if per_moment_sample_num is not None else [len(sample_indx)], # [t1, t2, ...]
            'labels': labels,  # [T,]
            'bbox': boxes,  # [T, 4], xyxy 这里是对的
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'orig_image_list': copy.deepcopy(imgs),
            'all_masks': all_masks,  # [vid_len, H, W]
            # 'objs_masks_dict': objs_masks_dict, # {'123': mask, boxes, norm_boxes, valid} # 仅在训练中使用
            'tgt_norm_bbox_list': tgt_norm_bbox_list, # [T,] # 仅在训练中使用，在test和val中为[]
            'frame_names': meta_copy['frames'],
            
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }

        # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
        # imgs, target = self._transforms(imgs, target)
        # imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]


        # # FIXME: may some data lost in stage1
        # # # FIXME: handle "valid", since some box may be removed due to random crop
        # if torch.any(target['valid'] == 1) or self.mode=='test':  # at leatst one instance 
        #     instance_check = True
        # else: # 这里如果有（随机裁剪导致）画面中没有实例，则会随机换一个实例作为输入
        #     idx = random.randint(0, self.__len__() - 1)

        

        if self.is_first:
            self.mock = copy.deepcopy(target)
            self.is_first = False
            
        return target


    def _st_ed_map_to_frames(self, moment_preds, vid_index):
        
        moment_preds = merge_overlapping_intervals(moment_preds) # 这里返回tensor，把重叠的部分合并
        
        moment_length = moment_preds[:, 1] - moment_preds[:, 0]
        norm_moment_length = moment_length / moment_length.sum() # 每个moment的比例
        per_moment_sample_num = torch.round(norm_moment_length * self.num_frames).to(dtype=torch.long) # 每个moment采样的帧数
        
        norm_moment_preds = moment_preds / (max(vid_index)+1) # 归一化，这里之前使用的长度不是150，所以没问题
        moment_preds = torch.round(norm_moment_preds * len(vid_index)).to(dtype=torch.long) # 放缩到vid_index的尺度下
        
        # 检查per_moment_sample_num的总和是否等于num_frames
        def adjust_to_sum_target(numbers, target_sum):  # 之后可能弃用对<img><vid>的情况
            # 将每个区间的采样帧数和调整到target_sum
            '''
            return
                numbers: [1,2,3,4]
                target_sum: 10
            '''
            total_sum = sum(numbers)
            if total_sum == target_sum:
                return numbers
            
            while total_sum > target_sum:
                max_value = max(numbers)
                max_index = numbers.index(max_value)
                numbers[max_index] -= 1
                total_sum -= 1
            
            while total_sum < target_sum:
                min_value = min(numbers)
                min_index = numbers.index(min_value)
                numbers[min_index] += 1
                total_sum += 1
                
            return numbers
        
        per_moment_sample_num = adjust_to_sum_target(per_moment_sample_num.tolist(), self.num_frames)
        assert sum(per_moment_sample_num) == self.num_frames, f'per_moment_sample_num should sum to {self.num_frames}'
        
        
        # create sample index
        sample_index = []
        for i, sample_num in enumerate(per_moment_sample_num): # 分段进行采样
            if sample_num == 0:
                continue
            # if sample_num=1 return start index
            st, ed = moment_preds[i][0], moment_preds[i][1]
            if ed >= len(vid_index): # 这里应该注意ed的边界条件
                ed = len(vid_index) - 1
            sample_idx = np.linspace(st, ed, sample_num, dtype=int) # 这里待验证
            sample_index.extend(vid_index[sample_idx])
    
        return np.array(sample_index), per_moment_sample_num


@METRICS.register_module()
class RVOSComputeMetrics(BaseComputeMetrics): # 这里应该计算每一对所有的预测（在cpu上，而不是在cuda上
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_iou = [] # [0.1, 0.2, 0, 1, ...]
        self.all_boundary_f = []
        # self.object_metrics = {}
        

    def __call__(self, preds_and_targets) -> Dict[str, Any]: # 这里设计为单个视频单个实例的计算，减少传输的成本
        return self.calculate_metric(preds_and_targets)
    
    def calculate_metric(self, preds_and_targets: list[torch.Tensor]): # 这里计算的是一个视频的
        pred_masks, gt_masks = preds_and_targets
        
        # 这里是sam2的，不合适，直接使用mevis的评估器
        # evaluator = Evaluator()
        # for pred_array, gt_array in zip(preds, gts):
        #     evaluator.feed_frame(mask=pred_array, gt=gt_array)
        # iou, boundary_f = evaluator.conclude()
        
        # 重新整理为二值图像的格式，而不是每个实例的格式
        iou = db_eval_iou(gt_masks, pred_masks).mean() # 这里比较慢，传入的是cpu数据
        boundary_f = db_eval_boundary(gt_masks, pred_masks).mean()
        
        self.all_iou.append(iou)
        self.all_boundary_f.append(boundary_f)
        
        # NOTE: please note iou only calculate for success target
    
    def get_all_metrics(self): # 这里直接返回当前处理的列表
        return self.all_iou, self.all_boundary_f
        # iou = np.array(self.all_iou).mean()
        # boundary_f = np.array(self.all_boundary_f).mean()
        # return iou, boundary_f
        
    def return_metrics(self, all_global_iou, all_global_boundary_f):
        global_j = np.array(all_global_iou).mean()
        global_f = np.array(all_global_boundary_f).mean()
        global_jf = (global_j + global_f) / 2
        
        return {
            'J': global_j,
            'F': global_f,
            'J&F': global_jf,
        }




if __name__ == '__main__':
    dataset = RVOSDataset(
        base_folder='/share_ssd/tianjirui/MeViS/train/', 
        num_frames=15,
        sampling_type='Uniform', # Uniform, Random, Dynamic
        dynamic_path='/share_ssd/tianjirui/MeViS/train/dynamic_path.json'
    )

    img, meta = dataset[11] # 11是两个mask