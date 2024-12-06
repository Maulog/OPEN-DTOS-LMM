import sys
import time
import logging
import json
import random
import requests
from PIL import Image
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import decord
from decord import VideoReader


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def read_img_general(img_path):
    if "s3://" in img_path:
        cv_img = read_img_ceph(img_path)
        # noinspection PyUnresolvedReferences
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as E:
            print(E)
            return Image.open("tmp.jpg").convert('RGB')

client = None


def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f"Please check image at {img_path}"
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    # noinspection PyUnresolvedReferences
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client(enable_mc=True)
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")
        
def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl_file(file_path: str) -> list:
    data_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed_line = json.loads(line.strip())
            data_list.append(parsed_line)

    return data_list        


def read_txt_file(file_path: str, return_str = False):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt_list = f.readlines()
        if not return_str:
            return txt_list
        else:
            return ''.join(txt_list)
        
        
        
def save_to_jsonl(data: list[dict], filename: str) -> None:
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item) + '\n'
                file.write(json_line)
        print(f"Data saved as JSONL file: {filename}")
    except IOError as e:
        print(f"Error saving JSONL file: {e}")
        
        
def save_to_json(data: list[dict], filename: str) -> None:
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data saved as JSON file: {filename}")
    except IOError as e:
        print(f"Error saving JSON file: {e}")
        
        
def pad_to_fixed_length(frms_list, fixed_length=100, padding_value=None):
    original_length = len(frms_list)
    padding = [padding_value] * fixed_length
    
    intervals = np.linspace(start=0, stop=fixed_length, num=original_length + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
    indices = [(x[0] + x[1]) // 2 for x in ranges] # 均匀填充
    
    for i, idx in enumerate(indices):
        padding[idx] = frms_list[i]
    
    return padding
        
def interpolate_missing_frames(frms_list): # 简单使用前后帧复制进行插值
    interpolated_frames = []
    prev_frame = None
    none_count = 0

    for frame in frms_list:
        if frame is None:
            none_count += 1
        else:
            if none_count > 0:
                if prev_frame is not None:
                    interpolated_frames.extend([prev_frame] * (none_count // 2))
                    none_count = none_count - none_count // 2
                    interpolated_frames.extend([frame] * none_count)
                else: # 视频开头都是none
                    interpolated_frames.extend([frame] * none_count)
                    
                interpolated_frames.append(frame)
                prev_frame = frame
                none_count = 0
            else: # 没有none
                interpolated_frames.append(frame)
                prev_frame = frame

    # 处理列表末尾的None值
    if none_count > 0:
        if prev_frame is not None:
            interpolated_frames.extend([prev_frame] * none_count)

    return interpolated_frames


        
        
def load_video(video_path, n_frms=100, height=-1, width=-1, sampling="uniform", clip_proposal=None, decoder_type = 'opencv',
               interpolated_frames=False):
    '''
    输入：视频路径，和需要采样的帧数
    返回：采样的帧，帧的索引，视频的帧率
    '''
    if decoder_type == 'decord':
        vr = VideoReader(uri=video_path, height=height, width=width)
        vlen = len(vr)
        n_frms = min(n_frms, vlen) # 选取设定帧数和视频长度的最小值
        fps = vr.get_avg_fps() 
        if clip_proposal is None:
            start, end = 0, vlen
        else:
            start, end = int(clip_proposal[0]*fps), int(clip_proposal[1]*fps) 
            # 如果指定了采样的片段，就按照片段的开始和结束时间来采样
            # clip_proposal是一个列表，里面是开始和结束时间（以秒为单位）
            if start < 0:
                start = 0
            if end > vlen:
                end = vlen

        intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1])) # 切成一个个小片段

        if sampling == 'random':
            indices = []
            for x in ranges:
                if x[0] == x[1]:
                    indices.append(x[0])
                else:
                    indices.append(random.choice(range(x[0], x[1])))
        elif sampling == 'uniform':
            # 取片段中间的帧
            indices = [(x[0] + x[1]) // 2 for x in ranges]

        elif sampling == "headtail": # 将视频分为两段，分别随机采样
            indices_h = sorted(random.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(random.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError
        
        if len(indices) < n_frms:
            rest = [indices[-1] for i in range(n_frms - len(indices))] # 如果采样的帧数不够，就重复最后一帧
            indices = indices + rest 
        # get_batch -> T, H, W, C
        
        frms = vr.get_batch(indices)
        frms = frms.asnumpy() # (T, H, W, C)
        frms_list = [Image.fromarray(frms[i]) for i in range(frms.shape[0])]
        
        vr = None

        return frms_list, indices, fps, vlen
    
    elif decoder_type == 'opencv':
        cap = cv2.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            raise ValueError(f"Cannot open video at {video_path}")
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        expect_sample_frms = n_frms
        n_frms = min(n_frms, vlen)
        if clip_proposal is None:
            start, end = 0, vlen
        else:
            start, end = int(clip_proposal[0]*fps), int(clip_proposal[1]*fps)
            if start < 0:
                start = 0
            if end > vlen:
                end = vlen

        intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))

        if sampling == 'random':
            indices = []
            for x in ranges:
                if x[0] == x[1]:
                    indices.append(x[0])
                else:
                    indices.append(random.choice(range(x[0], x[1])))
        elif sampling == 'uniform':
            indices = [(x[0] + x[1]) // 2 for x in ranges]
        elif sampling == "headtail":
            indices_h = sorted(random.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(random.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        frms_list = []
        # indices1 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frms_list.append(None)
                continue
                # while True:
                #     idx += 1
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                #     ret, frame = cap.read()
                #     if ret or idx >= vlen:
                #         break
            frms_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            # indices1.append(idx)
            
        cap = None

        if interpolated_frames:
            if vlen < expect_sample_frms:
                frms_list = pad_to_fixed_length(frms_list, fixed_length=expect_sample_frms, padding_value=None)
            frms_list = interpolate_missing_frames(frms_list)
            
        return frms_list, indices, fps, vlen

def load_images(image_files):
    images = []
    for image_file in image_files:
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        images.append(image)
    return images