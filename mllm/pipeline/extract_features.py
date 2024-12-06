import argparse
import os
import sys
import json
import re
import typing
import requests
from PIL import Image
from io import BytesIO
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent
import time
sys.path.append('/home/tianjirui/DTOS-LMM')

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig

from mllm.utils import (disable_torch_init, 
                        get_model_name_from_path, 
                        KeywordsStoppingCriteria, 
                        tokenizer_image_token,
                        )
from mllm.config.constants import *
from mllm.config import prepare_args

from mllm.models.builder import load_pretrained
from llava.mm_utils import process_images 
from mllm.dataset.utils import load_video, read_jsonl_file, read_txt_file, ImageGrid
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config
from llava.model.language_model.llava_llama import LlavaLlamaConfig
from llava.model.builder import prepare_config_for_eval
from mllm.models.builder.build_dtos import prepare_model_args
from mllm.models.dtos.dtos_base import DtosConfig, DtosLmmForCausalLM

key_folder = "ig100_vi20_sub4" # "pool_video_100"
image_grid = ImageGrid()

def check_video(frms_and_video_name, feat_path):
    frms, video_name = frms_and_video_name
    if all(element is None for element in frms[0]): # 此处记得修改
        with open(os.path.join(feat_path, key_folder, "invalid_videos.txt"), "a") as f:
            f.write(video_name + "\n")
        tqdm_instance.update(1)
        return False
    return True
    

def load_video_to_img(one_video_dataset, video_path):
    st = time.time()
    video_name = one_video_dataset['video']
    video_path = os.path.join(video_path, video_name)
    frms = load_video(video_path, decoder_type = 'opencv', n_frms = 120, sampling = 'uniform', 
                        clip_proposal = one_video_dataset['clip'], interpolated_frames=True)
    video_name = one_video_dataset['video'].rsplit('.', 1)[0]
    ed = time.time()
    print('load time:', ed - st)
    return (frms, video_name)
    
def extract_video_features(vision_encoder, image_processor, mm_projector, frms_and_video_name, save_path = None): # 单个视频
    st = time.time()
    frms, video_name = frms_and_video_name
    
    #### this code is 1+9 grid ###
    
    vid_ids, img_ids = image_grid.get_video_and_image_ig_id(np.arange(0, 100, 1), image_num=10, rough_img_num_sub_interval=9)
    
    img_frms = [frms[0][img_id] for img_id in img_ids]
    vid_frms = []
    for vid_id_interval in vid_ids:
        tmp_vid_frms = []
        for vid_id in vid_id_interval:
            tmp_vid_frms.append(frms[0][vid_id])
        vid_frm = image_grid.get_single_image_grid_col(tmp_vid_frms, image_rows=3, image_cols=3)
        vid_frms.append(vid_frm)

    ##############################

    #### this code is 4+4 grid ### 
    
    # vid_ids = image_grid.get_video_ig_id(np.arange(0, 120, 1), video_num=20, rough_img_num_sub_interval=6)
    
    # vid_frms = []
    # img_frms = []
    # for i, vid_id_interval in enumerate(vid_ids):
    #     tmp_ig_frms = []
    #     for vid_id in vid_id_interval:
    #         tmp_ig_frms.append(frms[0][vid_id])
    #     ig_frm = image_grid.get_single_image_grid(tmp_ig_frms, image_rows=3, image_cols=2)
        
    #     if i % 2 == 0:
    #         img_frms.append(ig_frm)
    #     else:
    #         vid_frms.append(ig_frm)
        
    ##############################
    
    def encode_images(frms_list):
        video_tensor = process_images(
            frms_list,
            image_processor,
            cfg.model_args
        ).to(vision_encoder.device, dtype=vision_encoder.dtype)
        
        vision_encoder.eval()
        mm_projector.eval()
        with torch.no_grad():
            features = vision_encoder(video_tensor) # siglip是[100, 729, 1152](729=27*27), CLIP是[100, 577, 1024]
            cls_features = mm_projector(features) # siglip是[100, 196(14*14), 4096]
                        
            cls_features = cls_features.to('cpu')
            torch.cuda.empty_cache()
            
            return cls_features
        
        
    image_cls_features = encode_images(img_frms)
    video_cls_features = encode_images(vid_frms)
            
    print(video_cls_features.shape, image_cls_features.shape)
            
    if save_path is not None:
        image_feat_save_path = os.path.join(image_feat_path, f"{video_name}.npy")
        video_feat_save_path = os.path.join(video_feat_path, f"{video_name}.npy")
        np.save(image_feat_save_path, image_cls_features.numpy())
        np.save(video_feat_save_path, video_cls_features.numpy())
        
    torch.cuda.empty_cache() 
    tqdm_instance.update(1)
    ed = time.time()
    print('extract time:', ed - st)
    
    return video_cls_features, image_cls_features

def load_and_extract_features(vision_encoder, image_processor, mm_projector, one_video_dataset, video_path, feat_path):
    # 能work应该是python没有真正多线程，在模型计算的时候还是单线程
    frms_and_video_name = load_video_to_img(one_video_dataset, video_path)
    if check_video(frms_and_video_name, feat_path):
        extract_video_features(vision_encoder, image_processor, mm_projector, frms_and_video_name, 
                               save_path = feat_path)

if __name__ == "__main__":
    
    dataset_type_list = ['train', 'validation', 'test'] # ['train', 'validation', 'test']
    dataset_name_list = ['didemo', 'qvhighlights', 'charadessta'] # ['activity_caption', 'didemo', 'qvhighlights', 'charadessta']
    '''
    'activity_caption' 
    'didemo'               
    'qvhighlights'         
    'charadessta'          
    'internvid_act'
    '''
    
    # borrowed from init_vlm
    cfg, training_args = prepare_args()
    config, kwargs = prepare_model_args(cfg.model_args)
    
    
    vision_tower, mm_projector = DtosLmmForCausalLM.load_visiontower_and_projection(config, **kwargs)
    image_processor = vision_tower.image_processor

    
    # # demo
    # video_path = "demo/musical.mov"
    # feat = extract_video_features(model.model.vision_tower, image_processor, video_path, save_path="demo/")

    
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            for dataset in cfg.data_args[dataset_type].cfgs:
                if dataset['feat_folder'].split('/')[-1] == dataset_name:
                    dataset_args = dataset
                    break
            
            print(f"Start extracting {dataset_name} {dataset_type} features.")
            
            video_path = dataset_args['video_folder']
            jsonl_path = dataset_args['filename']
            feat_path = dataset_args['feat_folder']
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)
                
            dataset_tree = read_jsonl_file(jsonl_path)
            # 遍历jsonl中的视频文件，输出特征并以.npy格式保存
            
            
            # 多线程,暂时没用
            # def worker(queue): # queue中暂时没用
            #     """从队列中提取结果并执行特征提取"""
            #     while True:
            #         item = queue.get()
            #         if item is None:  # 如果收到结束信号，则退出
            #             break
            #         extract_video_features(vision_tower, image_processor, item, save_path=feat_path)
            #         print('queue size:', queue.qsize())
            #         queue.task_done()

            # 初始化队列和线程池
            # result_queue = queue.Queue()
            # executor = ThreadPoolExecutor(max_workers=1)
            # worker_thread = threading.Thread(target=worker, args=(result_queue,))    
            # worker_thread.start()

            missing_videos = []
            video_name_set = set()
            image_feat_path = os.path.join(feat_path, key_folder, 'image')
            video_feat_path = os.path.join(feat_path, key_folder, 'video')
            os.makedirs(image_feat_path, exist_ok=True)
            os.makedirs(video_feat_path, exist_ok=True)
            existed_feats = os.listdir(image_feat_path)
            for one_video_dataset in dataset_tree: # 正常使用
                video_name = one_video_dataset['video'].rsplit('.', 1)[0]
                feat_name = video_name + '.npy' # 这里有修改
                if feat_name not in existed_feats and video_name not in video_name_set: # feat_name not in existed_feats
                    missing_videos.append(one_video_dataset)
                    video_name_set.add(video_name)
                    
                    
                    
            # invalid_videos = set(read_txt_file(os.path.join(feat_path, "pool_video_100", "invalid_videos.txt")))
            # for one_video_dataset in dataset_tree: # 用于挑选出invalid_videos进行重提特征
            #     video_name = one_video_dataset['video'].rsplit('.', 1)[0]
            #     if video_name in invalid_videos:
            #         continue
            #     else:
            #         missing_videos.append(one_video_dataset)
                    
            tqdm_instance = tqdm(total=len(missing_videos))
            
            

            # 提交任务到线程池
            executor = ThreadPoolExecutor(max_workers=3) # 不能太大比如6
            with executor:
                # 无queue版本
                futures = [
                    executor.submit(load_and_extract_features, vision_tower, image_processor, mm_projector,
                                    one_video_dataset, video_path = video_path, feat_path = feat_path
                                    ) for one_video_dataset in missing_videos # dataset_tree, # 用missing_videos是为了避免重复提取
                ]
                
                # # queue版本, 会导致加载较快（生产端快）特征提取器的速度慢（消费端慢）
                # futures = [
                #     executor.submit(load_video_to_img, one_video_dataset, video_path)
                #     for one_video_dataset in dataset_tree
                # ]
                # # 将完成的任务结果放入队列
                # for future in concurrent.futures.as_completed(futures):
                #     result = future.result()
                #     result_queue.put((result))

            # 当所有任务提交完毕后，向队列发送结束信号
            # result_queue.put(None)
            # worker_thread.join()
            tqdm_instance.close()

            print(f"{dataset_name} {dataset_type} task of extraction completed.")
    
    
    
        
    # # 单线程版本
    # for one_video_dataset in dataset_tree:
    #     feat = load_and_extract_features(model.model.vision_tower, image_processor, one_video_dataset,
    #                                      video_path = video_path, feat_path = feat_path)
    
    # print(f"{dataset_name} features extracted and saved in {feat_path}")