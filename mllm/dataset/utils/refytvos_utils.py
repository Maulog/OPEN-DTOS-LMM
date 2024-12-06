###########################################################################
# Created by: Xiaohongshu
# Email: clyanhh@gmail.com
# Copyright (c) 2023
###########################################################################

# modified from visa

import json
import logging
import numpy as np
import os
import os.path as osp
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import pickle

import pycocotools.mask as maskUtils

try:
    from .categories import ytvos_category_dict
    from .categories import davis_category_dict
except:
    from categories import ytvos_category_dict
    from categories import davis_category_dict
"""
This file contains functions to parse Refer-Youtube-VOS dataset of COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

def encode_anno_mask(frames, vid_len, img_folder, video, obj_id, anno_id, meta, ):
    anno_id_segm_list = list()
    for frame_idx in range(vid_len):
        frame_name = frames[frame_idx]
        mask_path  = os.path.join(str(img_folder), 'Annotations', video, frame_name + '.png')
        mask       = Image.open(mask_path).convert('P')
        mask       = np.array(mask, )
        mask       = (mask == obj_id) # 0, 1 binary
        segm       = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))) if mask.any() else None
        anno_id_segm_list.append(segm)
    assert len(anno_id_segm_list) == meta['length']
    assert len(anno_id_segm_list) == len(meta['frames'])
    return {str(anno_id): anno_id_segm_list}


def load_mevis_json(img_folder: str, mode: str):
    ann_file = os.path.join(img_folder, 'meta_expressions.json')
    mask_json = os.path.join(img_folder, 'mask_dict.json') # 这里进行判断！！
    
    # read expression data
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    vid_len = len(videos)

    metas = []
    for vid in videos: # 每个视频
        # vid_meta = subset_metas_by_video[vid]
        vid_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len = len(vid_frames)

        for exp_id, exp_dict in vid_data['expressions'].items(): # 视频中的每个表达
            # 修改这里，这里应该不需要切片这么细，这里应该返回不同采样策略下的数据
            # for frame_id in range(0, vid_len, self.num_frames): # num_frames是步长，均匀采样生成多个切片
            meta = {}
            meta['video'] = vid
            meta['exp'] = exp_dict['exp']
            meta['exp_id'] = exp_id
            if mode != 'test':
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
            else: # dummy data
                meta['obj_id'] = [-1]
                meta['anno_id'] = ['-1']
            # meta['frame_id'] = frame_id
            meta['category'] = 0
            meta['frames'] = vid_frames
            metas.append(meta)
    
    if mode != 'test':
        with open(mask_json) as fp:
            mask_dict = json.load(fp)
    else:
        mask_dict = None
        
    return metas, mask_dict


def load_refytvos_json(img_folder: str, ann_file: str, dataset_name, mode):
    """
    img_folder (str)    : path to the folder where 'Annotations' && 'JPEGImages' && 'meta.json' are stored.
    ann_file (str)      : path to the json file.
    """

    def prepare_metas():
        if mode == 'train':
            # read object information
            with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
                subset_metas_by_video = json.load(f)['videos']
            
            # read expression data
            with open(str(ann_file), 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            videos = sorted(list(subset_expressions_by_video.keys()))

            metas = []
            anno_count = 0  # serve as anno_id
            vid2metaid = defaultdict(list)
            for vid in videos: # 遍历每个视频
                vid_meta   = subset_metas_by_video[vid]
                vid_data   = subset_expressions_by_video[vid]
                vid_frames = sorted(vid_data['frames'])
                vid_len    = len(vid_frames)

                exp_id_list = sorted(list(vid_data['expressions'].keys()))
                for exp_id in exp_id_list: # 遍历每个表达
                    exp_dict            = vid_data['expressions'][exp_id]
                    meta                = {}
                    meta['video']       = vid
                    meta['exp']         = exp_dict['exp']
                    meta['obj_id']      = [0, ]  # Ref-Youtube-VOS only has one object per expression
                    meta['anno_id']     = [str(anno_count), ]
                    anno_count         += 1
                    meta['frames']      = vid_frames
                    meta['exp_id']      = exp_id
                    obj_id              = exp_dict['obj_id']
                    meta['obj_id_ori']  = int(obj_id)
                    meta['category']    = vid_meta['objects'][obj_id]['category']
                    meta['length']      = vid_len
                    metas.append(meta)
                    vid2metaid[vid].append(len(metas) - 1)
        else: # 这里有待检查(考虑直接传入mode，适配自己的，仿照mevis)
            # for some reasons the competition's validation expressions dict contains both the validation (202) & 
            # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
            # the validation expressions dict:
            # 明天改这里，（当移植youtube vos的时候看一看）!!!
            data = json.load(open(ann_file, 'r'))["videos"]
            videos = set(data.keys())
            # test_meta_file = ann_file.replace('valid/meta_expressions.json', 'test/meta_expressions.json')
            # test_data = json.load(open(test_meta_file, 'r'))["videos"]
            # test_videos = set(test_data.keys())
            # valid_videos = valid_test_videos - test_videos
            video_list = sorted([video for video in videos])
            # assert len(video_list) == 202, 'error: incorrect number of validation videos'
            metas = [] # list[dict], length is number of expressions
            vid2metaid = defaultdict(list)
            for video in video_list:
                expressions = data[video]["expressions"]
                expression_list = list(expressions.keys()) 
                num_expressions = len(expression_list)
                video_len = len(data[video]["frames"])

                # read all the anno meta
                for i in range(num_expressions):
                    meta = {}
                    meta["video"]    = video
                    meta["exp"]      = expressions[expression_list[i]]["exp"]
                    meta['obj_id'] = [-1]
                    meta['anno_id'] = ['-1']
                    meta["frames"]   = data[video]["frames"]
                    meta["exp_id"]   = expression_list[i]
                    meta['category'] = 0
                    meta['length']   = video_len
                    metas.append(meta)
                    vid2metaid[video].append(len(metas) - 1)

        return metas, vid2metaid


    mask_json           = os.path.join(img_folder, 'mask_dict.pkl')
    read_mask_from_json = osp.exists(mask_json)
    if read_mask_from_json:
        with open(mask_json, 'rb') as f:
            mask_dict = pickle.load(f)
    else:
        mask_dict = None  # need to be filled later, anno_id -> frame_id
    
    metas, vid2metaid = prepare_metas()
    return metas, mask_dict, vid2metaid


def load_refdavis_json(img_folder: str, ann_file: str, dataset_name: str, mode) :
    """
    img_folder (str)    : path to the folder where 'Annotations' && 'JPEGImages' && 'meta.json' are stored.
    ann_file (str)      : path to the json file.
    """

    def prepare_metas():
        # read object information
        with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        videos = sorted(list(subset_expressions_by_video.keys()))

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = defaultdict(list)
        for vid in videos: # 遍历每个视频
            vid_meta   = subset_metas_by_video[vid]
            vid_data   = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len    = len(vid_frames)

            exp_id_list = sorted(list(vid_data['expressions'].keys()))
            for exp_id in exp_id_list: # 遍历每个表达
                exp_dict            = vid_data['expressions'][exp_id]
                meta                = {}
                meta['video']       = vid
                meta['exp']         = exp_dict['exp']
                meta['obj_id']      = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id']     = [str(anno_count), ]
                anno_count         += 1
                meta['frames']      = vid_frames
                meta['exp_id']      = exp_id
                obj_id              = exp_dict['obj_id']
                meta['obj_id_ori']  = int(obj_id)
                meta['category']    = vid_meta['objects'][obj_id]['category']
                meta['length']      = vid_len
                metas.append(meta)
                vid2metaid[vid].append(len(metas) - 1)
                
        return metas, vid2metaid

    mask_json           = os.path.join(img_folder, 'mask_dict.pkl')
    read_mask_from_json = osp.exists(mask_json)
    if read_mask_from_json:
        with open(mask_json, 'rb') as f:
            mask_dict = pickle.load(f)
    else:
        mask_dict = None  # need to be filled later, anno_id -> frame_id
    
    metas, vid2metaid = prepare_metas()
    return metas, mask_dict, vid2metaid