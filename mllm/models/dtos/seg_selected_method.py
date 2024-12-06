from collections import Counter
import copy
import json
import logging
import os
import sys
import re
import types
from typing import Optional, List, Union, Tuple
import warnings
import deepspeed


import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou, nms
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig
from peft import PeftModel


def method1(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        nms_iou_threshold, # 超参
    ):
    '''
    (该方法适合检测非常精准的时候，没有添加剔除离群值)
    首先通过nms过滤重复框，然后选取box数最大的一帧作为最优帧（都可信减少漏检的可能性）
    如果多帧都是最大框数的，则选择iou和最小的帧（因为目标物比较分散，减少了遮挡带来的影响）
    '''
    selected_boxes_list = []
    selected_scores_list = []
    for boxes, scores, tgt_id in zip(all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx):
        if boxes is not None and scores is not None:
            indices = nms(boxes, scores, nms_iou_threshold)
            selected_boxes_list.append(boxes[indices])
            selected_scores_list.append(scores[indices])
            
    # max boxes num
    selected_boxes_num = [len(x) for x in selected_boxes_list]
    def find_max_indices(lst):
        max_value = max(lst)
        max_indices = [i for i, value in enumerate(lst) if value == max_value]
        return max_indices
    max_boxes_num_idx = find_max_indices(selected_boxes_num) # 首先找出
    
    # min iou sum
    if len(max_boxes_num_idx):
        min_iou_sum = None
        for idx in max_boxes_num_idx:
            selected_boxes = selected_boxes_list[idx]
            selected_scores = selected_scores_list[idx]
            selected_tgt_id = tgt_frm_idx[idx]
            if len(selected_boxes) > 1: # 待验证
                iou_matrix = box_iou(selected_boxes, selected_boxes)
                cur_iou_sum = torch.sum(iou_matrix) # 当前帧所有框的重合度，如果iou越大则证明越重合不能分开
            
            if min_iou_sum is None or min_iou_sum > cur_iou_sum: # init and min_iou selecting
                min_iou_sum = cur_iou_sum
                filter_boxes = selected_boxes
                filter_scores = selected_scores
                tgt_frame_idx = selected_tgt_id
    
    return [filter_boxes], [filter_scores], [selected_tgt_id]



def method2( # 与method3差不多，nms需要一个比较稳定的iou阈值
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    （该方法适合bbox比较少的场景，且有一部分离群框，nms_iou_threshold设置偏高一些，保留更多框）
    首先剔除离群值，即单独有检测框与其他所有的检测框的iou小于阈值overlap_iou_threshold，然后用nms去除重叠框（nms_iou_threshold）
    统计剩下每一帧中的框的数量，找到众数（稳定），如果有多个众数则选择最大众数的（因为找的全，在一帧中尽可能包含了多个目标），
    然后在众数中找到iou和最小的帧（最分散（适用于有一定精度））
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    # 剔除离群值
    for boxes, scores, tgt_id in zip(all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx): # 每一帧
        # 去除没有重合的离群框（用overlap_iou_threshold）
        boxes_num = len(boxes)
        iou_matrix = box_iou(boxes, boxes)
        overlap_mask = iou_matrix > overlap_iou_threshold # 大于阈值为true，认为重叠
        per_bbox_overlap_num = overlap_mask.sum(dim=1)-1 # 每个框与其他框重叠的数量 shape=[boxes_num]
        valid_mask = per_bbox_overlap_num > 0 # 用数量作为筛选，不用比例（因为会随着bbox的增加导致比例降低），有多个框重叠则认为有效
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool) # 如果全部都不重叠的框，则全部有效
            
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        # nms
        indices = nms(boxes, scores, nms_iou_threshold)
        
        selected_boxes_list.append(boxes[indices])
        selected_scores_list.append(scores[indices])
        
    selected_boxes_num = [len(x) for x in selected_boxes_list]
    def find_max_mode(numbers):
        counts = Counter(numbers)
        # 这里应该剔除数字0，看下是否危险，是否可能去除0导致整个为空，得确保必须有返回值
        if 0 in counts:
            del counts[0]  # 移除0
            if not counts: # 如果是全0则不存在这个变量
                counts[0] = 1
        max_count = max(counts.values())
        modes = [num for num, count in counts.items() if count == max_count]
        
        max_mode = max(modes) # 最大的众数，这里也可以取最小的（取决于模型的定位能力，定位能力好最大容易找全，定位能力差可以取最小防止干扰）
        indices = [i for i, num in enumerate(numbers) if num == max_mode]
        return max_mode, indices
    
    # min_mode_indices即最优帧的候选
    max_mode, max_mode_indices = find_max_mode(selected_boxes_num)
    min_iou_sum = None
    for mode_idx in max_mode_indices:
        selected_boxes = selected_boxes_list[mode_idx]
        selected_scores = selected_scores_list[mode_idx]
        iou_matrix = box_iou(selected_boxes, selected_boxes)
        cur_iou_sum = torch.sum(iou_matrix)
        if min_iou_sum is None or min_iou_sum > cur_iou_sum: 
            # (这里找的是偏后面的帧)
            # 如果是min_iou_sum,则倾向于找到两个目标比较分开的框（仅限于定位准的情况）
            # 如果是max_iou_sum，则倾向于找到目标重合比较多的框，更依赖于准度（因为很多obj是重叠的，适合找到聚在一起的obj，容易碰到目标物）
            min_iou_sum = cur_iou_sum
            filter_boxes = selected_boxes
            filter_scores = selected_scores
            selected_tgt_id = tgt_frm_idx[mode_idx]
            
    return [filter_boxes], [filter_scores], [selected_tgt_id]
            



def method3(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    （该方法适合bbox比较多的场景， nms_iou_threshold设置偏低，更大减少实际框）
    首先剔除离群值，即单独有检测框与其他所有的检测框的iou小于阈值overlap_iou_threshold，然后用nms去除重叠框（nms_iou_threshold）
    统计剩下每一帧中的框的数量，找到众数（稳定），如果有多个众数则选择最小众数的（因为在精度较低时较小众数意味着更容易被相信，
    较大众数往往容易框入其他目标），然后在众数中找到iou和最大的帧（最集中（适用于精度较低））
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    # 剔除离群值
    for boxes, scores, tgt_id in zip(all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx): # 每一帧
        # 去除没有重合的离群框（用overlap_iou_threshold）
        boxes_num = len(boxes)
        iou_matrix = box_iou(boxes, boxes)
        overlap_mask = iou_matrix > overlap_iou_threshold # 大于阈值为true，认为重叠
        per_bbox_overlap_num = overlap_mask.sum(dim=1)-1 # 每个框与其他框重叠的数量 shape=[boxes_num]
        valid_mask = per_bbox_overlap_num > 0 # 用数量作为筛选，不用比例（因为会随着bbox的增加导致比例降低），有多个框重叠则认为有效
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool) # 如果全部都不重叠的框，则全部有效
            
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        # nms
        indices = nms(boxes, scores, nms_iou_threshold)
        
        selected_boxes_list.append(boxes[indices])
        selected_scores_list.append(scores[indices])
        
    selected_boxes_num = [len(x) for x in selected_boxes_list]
    def find_min_mode(numbers):
        counts = Counter(numbers)
        # 这里应该剔除数字0，看下是否危险，是否可能去除0导致整个为空，得确保必须有返回值
        if 0 in counts:
            del counts[0]  # 移除0
            if not counts: # 如果是全0则不存在这个变量
                counts[0] = 1
        max_count = max(counts.values())
        modes = [num for num, count in counts.items() if count == max_count]
        
        min_mode = min(modes) # 最大的众数，这里也可以取最小的（取决于模型的定位能力，定位能力好最大容易找全，定位能力差可以取最小防止干扰）
        indices = [i for i, num in enumerate(numbers) if num == min_mode]
        return min_mode, indices
    
    # min_mode_indices即最优帧的候选
    min_mode, min_mode_indices = find_min_mode(selected_boxes_num)
    max_iou_sum = None
    for mode_idx in min_mode_indices:
        selected_boxes = selected_boxes_list[mode_idx]
        selected_scores = selected_scores_list[mode_idx]
        iou_matrix = box_iou(selected_boxes, selected_boxes)
        cur_iou_sum = torch.sum(iou_matrix)
        if max_iou_sum is None or max_iou_sum < cur_iou_sum: 
            # (这里找的是偏后面的帧)
            # 如果是min_iou_sum,则倾向于找到两个目标比较分开的框（仅限于定位准的情况）
            # 如果是max_iou_sum，则倾向于找到目标重合比较多的框，更依赖于准度（因为很多obj是重叠的，适合找到聚在一起的obj，容易碰到目标物）
            max_iou_sum = cur_iou_sum
            filter_boxes = selected_boxes
            filter_scores = selected_scores
            selected_tgt_id = tgt_frm_idx[mode_idx]
            
    return [filter_boxes], [filter_scores], [selected_tgt_id]
            
            
def method4(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    (设置一个偏高的nms_iou_threshold，保留更多框，适合bbox比较多的场景)
    直接返回等于众数的帧都作为最优帧，如果有多个众数则选择最大众数(检测比较全)
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    # 剔除离群值
    for boxes, scores, tgt_id in zip(all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx): # 每一帧
        # 去除没有重合的离群框（用overlap_iou_threshold）
        boxes_num = len(boxes)
        iou_matrix = box_iou(boxes, boxes)
        overlap_mask = iou_matrix > overlap_iou_threshold # 大于阈值为true，认为重叠
        per_bbox_overlap_num = overlap_mask.sum(dim=1)-1 # 每个框与其他框重叠的数量 shape=[boxes_num]
        valid_mask = per_bbox_overlap_num > 0 # 用数量作为筛选，不用比例（因为会随着bbox的增加导致比例降低），有多个框重叠则认为有效
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool) # 如果全部都不重叠的框，则全部有效
            
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        # nms
        indices = nms(boxes, scores, nms_iou_threshold)
        
        selected_boxes_list.append(boxes[indices])
        selected_scores_list.append(scores[indices])
        
    selected_boxes_num = [len(x) for x in selected_boxes_list]
    def find_max_mode(numbers):
        counts = Counter(numbers)
        # 这里应该剔除数字0，看下是否危险，是否可能去除0导致整个为空，得确保必须有返回值
        if 0 in counts:
            del counts[0]  # 移除0
            if not counts: # 如果是全0则不存在这个变量
                counts[0] = 1
        max_count = max(counts.values())
        modes = [num for num, count in counts.items() if count == max_count]
        
        max_mode = max(modes) # 最大的众数，这里也可以取最小的（取决于模型的定位能力，定位能力好最大容易找全，定位能力差可以取最小防止干扰）
        indices = [i for i, num in enumerate(numbers) if num == max_mode]
        return max_mode, indices
    
    # min_mode_indices即最优帧的候选
    max_mode, max_mode_indices = find_max_mode(selected_boxes_num)
    
    # 使用所有的众数目标书太多会超显存
    # 这里只取三个分数最高的帧作为最优帧
    if len(max_mode_indices) > 2: # 只筛选出对应
        max_scores = [torch.max(selected_scores_list[ind]) for ind in max_mode_indices]
        topk_scores, topk_indices = torch.topk(torch.stack(max_scores), 2)
        # 过滤分数最高的三帧
        topk_indices, _ = torch.sort(topk_indices)
        max_mode_indices = [max_mode_indices[ind] for ind in topk_indices]
    
    filter_boxes = [selected_boxes_list[ind] for ind in max_mode_indices]
    filter_scores = [selected_scores_list[ind] for ind in max_mode_indices]
    selected_tgt_id = [tgt_frm_idx[ind] for ind in max_mode_indices]
    
    return filter_boxes, filter_scores, selected_tgt_id

# ablation study
def top1_frm_method(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    找到所有候选框中分数最高的一帧作为最优帧
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    max_score_list = [max(scores) for scores in all_frames_scores_list]
    max_score = max(max_score_list)
    max_score_frm_idx = max_score_list.index(max_score)
    
    boxes = all_frames_boxes_list[max_score_frm_idx]
    scores = all_frames_scores_list[max_score_frm_idx]
    selected_tgt_id_list = [tgt_frm_idx[max_score_frm_idx]]
    
    # nms
    indices = nms(boxes, scores, nms_iou_threshold)
    selected_boxes_list.append(boxes[indices])
    selected_scores_list.append(scores[indices])
    
    return selected_boxes_list, selected_scores_list, selected_tgt_id_list


def middle_frm_method(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    最中间的一帧作为最优帧
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    middle_frm_idx = len(tgt_frm_idx) // 2
    
    boxes = all_frames_boxes_list[middle_frm_idx]
    scores = all_frames_scores_list[middle_frm_idx]
    selected_tgt_id_list = [tgt_frm_idx[middle_frm_idx]]
    
    # nms
    indices = nms(boxes, scores, nms_iou_threshold)
    selected_boxes_list.append(boxes[indices])
    selected_scores_list.append(scores[indices])
    
    return selected_boxes_list, selected_scores_list, selected_tgt_id_list



def random_frm_method(
        all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
        overlap_iou_threshold, nms_iou_threshold, # 超参
    ):
    '''
    随机选择一帧作为最优帧
    '''
    selected_boxes_list = []
    selected_scores_list = []
    
    random_frm_idx = np.random.randint(len(tgt_frm_idx))
    
    boxes = all_frames_boxes_list[random_frm_idx]
    scores = all_frames_scores_list[random_frm_idx]
    selected_tgt_id_list = [tgt_frm_idx[random_frm_idx]]
    
    # nms
    indices = nms(boxes, scores, nms_iou_threshold)
    selected_boxes_list.append(boxes[indices])
    selected_scores_list.append(scores[indices])
    
    return selected_boxes_list, selected_scores_list, selected_tgt_id_list