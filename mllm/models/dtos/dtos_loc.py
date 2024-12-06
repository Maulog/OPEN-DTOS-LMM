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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig
from peft import PeftModel

# from mllm.models.vila.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaModel
from mllm.dataset.utils.image_grid import ImageGrid
from mllm.models.dtos.dtos_base import DtosLmmForCausalLM, DtosConfig, PartialTrainableEmbedding, SpecialTokenClassifier
from mllm.models.dtos.dtos_loss import CycleLoss, language_modeling_loss, no_special_token_language_modeling_loss, compute_rec_loss, rec_binary_loss
from mllm.models.moment_detr.matcher import HungarianMatcher
from mllm.models.moment_detr.postprocessing_moment_detr import PostProcessorDETR
from mllm.utils.box_ops import box_iou, generalized_box_iou
from mllm.config.constants import *
from mllm.utils.span_utils import generalized_temporal_iou

torch.autograd.set_detect_anomaly(True) # 用于检查定位反向传播时报的错

def prepare_inputs_for_generation( # 暂时没用
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2] # 获取第一层和第一个头的序列长度

        # Some generation methods already pass only the last input ID
        if inputs_embeds.shape[1] > past_length: 
            remove_prefix_length = past_length
        else: # 有的可能生成多个新的token，只需要保留最后一个token作为新的输入即可
            # Default to old behavior: keep only final ID
            remove_prefix_length = inputs_embeds.shape[1] - 1

        # input_ids = input_ids[:, remove_prefix_length:] # 此处用于去除过去重复的部分
        inputs_embeds = inputs_embeds[:, remove_prefix_length:, :]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None: # 由attentionmask动态生成position_ids
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1 # 沿最后一个维度（seq_len）累加，-1是因为0开始不是1开始
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # 修改这里，每次都使用emb作为输入，而不是只有第一次使用emb作为输入
    if inputs_embeds is not None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
  
      

    
class CategoryDimensionFc(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, dropout_rate=0.5, norm='LN'):
        super(CategoryDimensionFc, self).__init__()

        self.layers = nn.ModuleList()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size_list[0]),
            # nn.BatchNorm1d(hidden_size_list[0]), # TODO 特殊情况 hidden_size_list[0]
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.layers.append(self.input_layer)
        if len(hidden_size_list) > 1:
            for i in range(1, len(hidden_size_list)):
                self.hidden_layer = nn.Sequential(
                    nn.Linear(hidden_size_list[i-1], hidden_size_list[i]),
                    # nn.BatchNorm1d(hidden_size_list[i]), # TODO 目前最好的版本需要带这个 TODO 特殊情况 hidden_size_list[i]
                    nn.LayerNorm(hidden_size_list[i]),
                    nn.Dropout(dropout_rate), # 目前最好的版本需要带这个
                    nn.ReLU() # 目前最好的版本需要带这个
                )
                self.layers.append(self.hidden_layer)
                del self.hidden_layer

        self.output_layer = nn.Linear(hidden_size_list[-1], output_size)
        self.layers.append(self.output_layer)

    def forward(self, cat_features):
        for layer in self.layers:
            cat_features = layer(cat_features)
        return cat_features
    
    

class DtosForLocLM(DtosLmmForCausalLM):
    def __init__(self, config: DtosConfig, *args, **kwargs):
        super(DtosForLocLM, self).__init__(config, *args, **kwargs) # 直接使用encode_input     
        
        
        self.rec_decoder = CategoryDimensionFc(
            self.config.hidden_size, 
            [
                self.config.hidden_size,#*2,
                self.config.hidden_size,#*2,
                # self.config.hidden_size,#*2,
            ], 
            self.config.rec_dim, 
            dropout_rate=0.5
        )
        
        # self.rec_decoder = nn.Sequential( # 将token映射到[start,end]的维度
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size, self.config.rec_dim) 
        # )
        
        self.rec_encoder = nn.Sequential( # 将[start,end]映射到hidden_size的维度
            nn.Linear(self.config.rec_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size) 
        )
        
        self.cycle_loss = CycleLoss(self.rec_encoder, self.rec_decoder)
        
        self.rec_score_head = nn.Linear(self.config.hidden_size, 2)
        self.background_label = 1
        self.foreground_label = 0
        
        self.image_grid = ImageGrid()
        
        self.matcher = HungarianMatcher(
            cost_class=self.config.cost_class, 
            cost_span=self.config.cost_span, 
            cost_giou=self.config.cost_giou, 
            span_loss_type=self.config.span_loss_type, 
            # max_v_l=75 # 没用
        )
        
        self.post_processor = PostProcessorDETR(
            clip_length=1, 
            move_window_method="left",
            process_func_names=("clip_window_l", "clip_ts")
        ) # 这里的clip_length是最小时间分辨率
        self.topk = 5
        self.score_threshold = 0.5
        
        self.gloabal_step = 0
        
        self.post_init()
        
    def build_rec_token_projector(self, new_token_num, vocab_size, hidden_dim, init_weight = None, from_pretrained = False):
        self.rec_token_projector = PartialTrainableEmbedding(
            new_token_num, vocab_size, hidden_dim, self.get_input_embeddings(), init_weight
        )
        return self.rec_token_projector
            
    def get_rec_token_projector(self):
        return self.rec_token_projector
    
    def build_rec_token_classifier(self, hidden_dim):
        self.rec_token_classifier = SpecialTokenClassifier(hidden_dim)
        return self.rec_token_classifier
    
    def get_rec_token_classifier(self):
        return self.rec_token_classifier
        
        
    def lm_head_seperately(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rec_token_classifier = self.get_rec_token_classifier()
        mask, prob_orig= rec_token_classifier(hidden_states)
        prob = prob_orig[mask]
        # seperate
        loc_hidden_states = hidden_states[mask]
        orig_hidden_states = hidden_states[~mask]
        # 将mask对应的部分生成为<rec>的ids
        orig_logits = self.get_output_embeddings()(orig_hidden_states)
        # return orig_logits, loc_hidden_states, mask

        loc_logits = torch.ones(
            (loc_hidden_states.shape[0], orig_logits.shape[-1]), 
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        loc_logits = loc_logits * (1 - prob)[:, None]/ (orig_logits.shape[-1] - 1) # 给负样本均分概率
        rec_id = self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN)
        loc_logits[:, rec_id] = prob # 构造onehot编码作为标签
        
        # combine
        # mask = mask.squeeze(0)
        ret = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1], orig_logits.shape[-1]),
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        ret[mask] = loc_logits
        ret[~mask] = orig_logits
        return ret, prob_orig
        
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length
        
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def prepare_inputs_labels_for_multimodal( # 该函数将分散的输入图片、视频和文本拼接在一起
        self, input_ids, position_ids, attention_mask, past_key_values, labels, 
        im_src, vi_src, im_feat, vi_feat, norm_timestamps, reverse_list
    ): # 这里应该是用不到vi_src的信息，直接由im_src计算得到
        vision_tower = self.get_vision_tower()
        
        if im_feat is not None: # 应该在此处提取视频信息，和对应的图片信息
            video_features = vi_feat
            image_features = im_feat
        if im_src is not None:
            # handle different image dtypes for packing
            if type(im_src) is list:
                im_src = torch.cat(im_src, dim=0)
            elif im_src.ndim == 5:  # batch_size x seq_len x image_channels
                im_src = im_src.flatten(0, 1)
            with torch.no_grad():
                im_src = im_src.to(device=self.device,dtype=self.dtype)
                vi_src = vi_src.to(device=self.device,dtype=self.dtype)
                n_images = (input_ids[0] == self.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)).sum() # 对话中图片的总数
                n_videos = (input_ids[0] == self.convert_tokens_to_ids(DEFAULT_VI_START_TOKEN)).sum() # 对话中视频的总数
                
                image_features = self.encode_images_low_usage(im_src).to(self.device) # 并返回视频
                video_features = self.encode_images_low_usage(vi_src).to(self.device)
                assert n_images == image_features.shape[0], 'The number of images token must be equal to the number of image features'
                assert n_videos == video_features.shape[0], 'The number of videos token must be equal to the number of video features'
            
        if image_features is None and video_features is None: # 之后可改成只图像或者只视频的
            raise ValueError("Image features or video features must be provided.")
            

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids_copy = input_ids.clone()
        
        # kentang-mit@: Otherwise tokenizer out of bounds. Embeddings of image tokens will not be used.
        
        # 这里需要再考虑一下，因为送入转换的是否要包括label
        # input_ids_copy[input_ids_copy == self.convert_tokens_to_ids(IMAGE_PLACEHOLDER)] = 0 # 好像没用
        # input_ids_copy[input_ids_copy == self.convert_tokens_to_ids(VIDEO_PLACEHOLDER)] = 0
        # input_embeds = self.llm.model.embed_tokens(input_ids_copy)
        
        input_embeds = self.rec_token_projector(input_ids_copy) # 用新写的
        rec_ids = self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN)
        rec_ids_mask = (input_ids_copy == rec_ids) & (labels==-100) # 筛选出<rec>的位置和系统的提问部分
        
        # 此处将<rec>变为嵌入拼接到input_embeds中，放在这里是因为generate也可以用
        # reverse_timestamps = [timestamps[i] for i, reverse in enumerate(reverse_list) if reverse]
        reverse_norm_timestamps = [norm_timestamps[i] for i, reverse in enumerate(reverse_list) if reverse]
        rec_feat_list = []
        for rts in reverse_norm_timestamps:
            rec_feat = self.rec_encoder(rts)
            rec_feat_list.append(rec_feat)
        if len(rec_feat_list) == 0: # 本轮对话中没有reverse该有的<rec>,但是本轮中的回答可能会有<rec>
            rec_feat_tensor = torch.zeros((0, self.config.hidden_size), device=input_embeds.device, dtype=input_embeds.dtype) # dummy
        else:
            rec_feat_tensor = torch.cat(rec_feat_list, dim=0)
        assert input_embeds[rec_ids_mask].shape == rec_feat_tensor.shape, 'The number of <rec> asked must be equal to the reversed timestamp'
        input_embeds[rec_ids_mask] = rec_feat_tensor # 检查debug中的这两个变量
        

        # 此部分代码用于筛选出有效token，通过attention_mask进行布尔索引
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_video_idx = 0

        # Note (kentang-mit@): image start / end is not implemented here to support pretraining.
        if getattr(self.config, "mm_use_im_start_end", False):
            # 此处写使用im_start_end的逻辑
            for batch_idx, cur_input_ids in enumerate(input_ids): # 每一个batch分开处理
                cur_input_ids = input_ids[batch_idx]
                num_images = (cur_input_ids == self.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)).sum() # 图片的总数
                num_videos = (cur_input_ids == self.convert_tokens_to_ids(DEFAULT_VI_START_TOKEN)).sum() # 视频的总数
                
                cur_input_embeds = input_embeds_1[batch_idx]
                cur_labels = labels[batch_idx]
                
                img_st_indices = torch.where(
                    cur_input_ids == self.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
                )[0].tolist()
                img_ed_indices = torch.where(
                    cur_input_ids == self.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
                )[0].tolist()
                assert len(img_st_indices) == len(img_ed_indices)
                vid_st_indices = torch.where(
                    cur_input_ids == self.convert_tokens_to_ids(DEFAULT_VI_START_TOKEN)
                )[0].tolist()
                vid_ed_indices = torch.where(
                    cur_input_ids == self.convert_tokens_to_ids(DEFAULT_VI_END_TOKEN)
                )[0].tolist()
                assert len(vid_st_indices) == len(vid_ed_indices)
                
                
                for i in range(num_videos): # 把对应位置替换为提取的特征
                    cur_input_embeds[vid_st_indices[i]+1:vid_ed_indices[i]] = video_features[i]
                    cur_labels[vid_st_indices[i]+1:vid_ed_indices[i]] = IGNORE_INDEX
                for i in range(num_images):
                    cur_input_embeds[img_st_indices[i]+1:img_ed_indices[i]] = image_features[i]
                    cur_labels[img_st_indices[i]+1:img_ed_indices[i]] = IGNORE_INDEX
                    
                    
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                
            
        else: # 使用<image>和<video>的token，video添加有问题，暂不使用
        
            # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
            for batch_idx, cur_input_ids in enumerate(input_ids): # 每一个batch分开处理
                cur_input_ids = input_ids[batch_idx]
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 图片的总数
                if num_images == 0: # 没有<image> token但是输入了图片并提取了特征，则直接选取第一个特征拼接到最后
                    cur_image_features = image_features[0]
                    cur_input_embeds_1 = input_embeds_1[batch_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    # kenang-mit@: we do not have placeholdr image for text-only data now.
                    # cur_image_idx += 1
                    continue
                
                ## 图片逐个拼接
                cur_input_embeds = input_embeds_1[batch_idx]
                image_token_indices = ( # 返回图像对应的索引（-1+index+ids_shape）
                    [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                )
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                cur_input_embeds_no_im = []
                for i in range(len(image_token_indices) - 1): # 根据<image>将对话切分为多个片段，不包含<image> token
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                    cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_new_input_embeds = []
                cur_new_labels = []
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append( # 将图片部分的标签设置为IGNORE_INDEX
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

                cur_new_input_embeds = torch.cat(cur_new_input_embeds) # 这里要改
                cur_new_labels = torch.cat(cur_new_labels)
                
                
                ## 视频逐个拼接，不能直接复用代码，因为label和index会在添加了图片后发生变化，使原来的ids和labels对应不上
                '''
                num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum() # 视频的总数
                video_token_indices = ( # 返回视频对应的索引（-2+index+ids_shape）
                    [-1] + torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                )
                cur_input_ids_novi = []
                cur_labels = labels[batch_idx]
                cur_labels_novi = []
                cur_input_embeds_no_vi = []
                for i in range(len(video_token_indices) - 1): # 根据<image>将对话切分为多个片段，不包含<image> token
                    cur_input_ids_novi.append(cur_input_ids[video_token_indices[i] + 1 : video_token_indices[i + 1]])
                    cur_labels_novi.append(cur_labels[video_token_indices[i] + 1 : video_token_indices[i + 1]])
                    cur_input_embeds_no_vi.append(cur_input_embeds[video_token_indices[i] + 1 : video_token_indices[i + 1]])
                split_sizes = [x.shape[0] for x in cur_labels_novi]
                cur_new_input_embeds = []
                cur_new_labels = []
                for i in range(num_videos + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_vi[i])
                    cur_new_labels.append(cur_labels_novi[i])
                    if i < num_videos:
                        cur_video_features = video_features[cur_video_idx]
                        cur_video_idx += 1
                        cur_new_input_embeds.append(cur_video_features)
                        cur_new_labels.append( # 将图片部分的标签设置为IGNORE_INDEX
                            torch.full(
                                (cur_video_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                '''

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                raise NotImplementedError

        # 如果超过最大长度进行截断
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        tokenizer_model_max_length = 8192 # 有风险的，之前可能并没有训练过
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds): # 默认值4096
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds] # 对超过长度的输入进行截断并形成一个列表
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds) # 找出所有batch中最长的序列长度
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 填充pad数据
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)): # 遍历填充每个批次
            cur_len = cur_new_embed.shape[0]
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left": # 左填充
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0: # 构造相关的标签、掩码、位置编码
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else: # 右填充
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values, # past_key_values在该函数中并没有处理
            new_input_embeds,
            new_labels,
        )
    
    def forward(self,
        input_ids: torch.LongTensor = None,
        im_src: Optional[torch.FloatTensor] = None,
        vi_src: Optional[torch.FloatTensor] = None,
        im_feat: Optional[torch.FloatTensor] = None,
        vi_feat: Optional[torch.FloatTensor] = None,
        timestamps: Optional[List[torch.FloatTensor]] = None,
        norm_timestamps: Optional[List[torch.FloatTensor]] = None,
        reverse_list: Optional[List[torch.FloatTensor]] = None,
        captions: Optional[List[str]] = None,
        clip_length: Optional[float] = None,
        data_dict: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                labels, 
                im_src, 
                vi_src, 
                im_feat, 
                vi_feat, 
                norm_timestamps,
                reverse_list # 用于记录正反
            )
        # Note (kentang-mit@): we have a unit test for this function.
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data( # 将长度相似的数据组合在一起形成新的批次
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        outputs = self.llm.forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        # return outputs
        last_hidden_states = outputs['hidden_states'][-1]
        loss = 0

        
        # logit为经过lm_head的输出
        # logits, mask_prob = self.lm_head_seperately(last_hidden_states)
        logits = outputs.logits # 看下output输出
        
        answer_mask = new_labels != IGNORE_INDEX       # 标签中对答案的掩码
        pred_logits = logits[answer_mask].squeeze(0)   # 预测的答案部分

        available_flag = True
        if labels is not None: 
            lang_model_loss = language_modeling_loss(logits.squeeze(0), new_labels.squeeze(0), len(self.tokenizer))
            # enhance the language without special tokens
            aug_lang_model_loss = no_special_token_language_modeling_loss(
                logits.squeeze(0), new_labels.squeeze(0), len(self.tokenizer), 
                self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN))
            
            # sifan nb!!!
            loss += lang_model_loss + aug_lang_model_loss*2 # 这里暂时没加入信息统计
            
            # binary_loss = rec_binary_loss(mask_prob[answer_mask], new_labels[answer_mask], self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN))
            # loss += binary_loss # 二分类没做移位！！！

        
        # rec_loss
        rec_loss = 0
        iou_loss = []
        error_loss = []
        l1_loss = []
        label_loss = []
        cycle_loss = []
        if norm_timestamps is not None and len(norm_timestamps) > 0: # 有时间戳的情况下
            pred_ids = torch.argmax(pred_logits, dim=-1)
            pred_text = self.decode_generate_ids(pred_ids)
            all_pred_ids = logits.argmax(-1).squeeze(0)
            
            ## 单元测试，测试下面的功能是否能正确运行
            # pred_ids = new_labels[answer_mask].squeeze(0)
            # pred_text = self.decode_generate_ids(pred_ids)
            # all_pred_ids = new_labels.squeeze(0)
            # print(pred_text) # 调试用
            ##
            
            def find_subseq_idx(ids, subseq): # 使用卷积查找完全一致的子字符串
                ids = ids.to(dtype=torch.float32)
                subseq = torch.tensor(subseq, dtype=torch.float32, device=ids.device)
                
                ids = ids.clone().detach().requires_grad_(False)
                subseq = subseq.clone().detach().requires_grad_(False)
                
                ids_unsqueezed = ids.unsqueeze(0).unsqueeze(0)
                target_unsqueezed = subseq.unsqueeze(0).unsqueeze(0)

                conv_result = F.conv1d(ids_unsqueezed, target_unsqueezed)
                matches = (conv_result == torch.sum(subseq**2)).nonzero(as_tuple=True)[2]
                return matches # 返回值是对应的ids开头的索引
            
            
            pred_mask = new_labels.squeeze(0) != IGNORE_INDEX
            rec_token_mask = (all_pred_ids == self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN)) & pred_mask # 这里筛选出预测部分的<rec>
            all_pred_ids_without_ignore = all_pred_ids.clone().detach()
            all_pred_ids_without_ignore[~pred_mask] = IGNORE_INDEX
            
            # 此处应该小心，应仔细检查解码的ids！！！！并且此处应该忽略all_pred_ids中被掩码的部分
            start_delim = ' pinpoint ten video moments' # ' locate these moments' # 这里长的可能会检测不到，怀疑超过了数据的长度
            end_delim = ' yields detailed answers' # ' my detailed answers' #    279,   2835
            start_ids = self.tokenizer(start_delim)['input_ids']
            end_ids = self.tokenizer(end_delim)['input_ids']
            rec_st_token_idx = find_subseq_idx(all_pred_ids_without_ignore, start_ids) # 不统计前面忽略的部分
            rec_ed_token_idx = find_subseq_idx(all_pred_ids_without_ignore, end_ids)
            
            rec_token_idx = torch.where(rec_token_mask)[0]
            
            rec_token_num = len(rec_token_idx)
            rec_st_token_num = len(rec_st_token_idx)
            rec_ed_token_num = len(rec_ed_token_idx)

            
            rec_id = self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN)
            
            no_reverse_num = reverse_list.count(False)
            # no_reverse_norm_timestamps = [norm_timestamps[i] for i, reverse in enumerate(reverse_list) if not reverse]
            
            
            available_list = [0]*len(captions) # 用来保存每个预测之间的<rec>数量，这里待修改，应该根据reverse_list判断
            if rec_st_token_num == rec_ed_token_num == no_reverse_num and rec_st_token_num and rec_token_num: # 有相同数量的<start>和<end>，且不为0
                # 过滤只取<rec_start>和<rec_end>之间的<rec>部分
                pred_target_list = [] # 此处用于更新available_list和rec_token_mask，防止出现<end><rec>这种情况
                st_ed_patten = torch.zeros_like(all_pred_ids, dtype=torch.bool, device=all_pred_ids.device)
                cnt = 0
                for i in range(len(captions)):
                    if not reverse_list[i]: # 正常推时间戳
                        st_idx, ed_idx = rec_st_token_idx[cnt], rec_ed_token_idx[cnt] # 计算rec_start和rec_end之间的<rec>数量
                        st_ed_patten[st_idx:ed_idx+1] = True
                        sub_interval = all_pred_ids[st_idx:ed_idx+1]
                        pred_target_list.append((sub_interval == rec_id).sum().item()) # 统计每个caption的<rec>数量
                        cnt += 1
                    else: # 推描述
                        pred_target_list.append(0)
                assert cnt == rec_st_token_num, 'The number of <rec> should be equal to the number of <rec_start> and <rec_end>'
                rec_token_mask = rec_token_mask & st_ed_patten # 只取<rec_start>和<rec_end>之间的<rec>部分
                available_list = pred_target_list
                
            rec_hidden_states = last_hidden_states.squeeze(0)[rec_token_mask] # 此处生成<rec>的hs
            
            #（这里不需要再次检查，因为如果出现推描述的时候推出了<rec>会造成卡死，只需要保证available_list是无误的就可以）
            # assert rec_hidden_states.shape[0] == sum(available_list), 'rec_hs must be equal to the number of available <rec>' 

            offset = 0
            for i in range(len(captions)):
                # 在当前caption下
                timestamp_labels, caption = norm_timestamps[i], captions[i] # label
                cur_rec_num = available_list[i]
                
                
                cur_rec_hs = rec_hidden_states[offset:offset+cur_rec_num]
                pred_rec = self.rec_decoder(cur_rec_hs)
                # pred_rec = F.sigmoid(pred_rec) # 限制了输出范围后，loss范围比较正常
                score = self.rec_score_head(cur_rec_hs)
                offset += cur_rec_num
                
                # circle_loss
                label_embed = self.rec_encoder(timestamp_labels)
                one_cycle_loss = self.cycle_loss(pred_rec, cur_rec_hs, label_embed, timestamp_labels) # 这里为了方便用的都是全部预测值，和全部标签值（无论正确与否）
                one_cycle_loss = torch.tensor(0, dtype=one_cycle_loss.dtype, device=one_cycle_loss.device) # 这里禁用
                
                # error_loss
                error_mask = pred_rec[:, 0] >= pred_rec[:, 1] # 无效的预测的掩码
                error_pred_rec = pred_rec[error_mask]
                # 此损失用于惩罚无效的预测，用求和是防止分母为0，*10是为了放大错误的惩罚
                one_error_loss = (error_pred_rec[:, 0] - error_pred_rec[:, 1]).sum()*5 
                
                correct_pred_rec = pred_rec[~error_mask]
                correct_pred_score = score[~error_mask]
                if len(correct_pred_rec) == 0: # 无有效预测，则构造有效的结果防止下面程序卡住（当能成功预测后，可能只会进前几次）
                    available_list[i] = 0 # 不计算后续的iou l1 label loss # 可能这有问题
                    correct_pred_rec = torch.tensor([[0,1]], dtype=correct_pred_rec.dtype, device=correct_pred_rec.device,
                                                   requires_grad=False).detach() # 不是detach的问题
                    correct_pred_score = torch.tensor([[1,3]], dtype=correct_pred_score.dtype, device=correct_pred_score.device,
                                                    requires_grad=False).detach()
                    

                ##### 当前任务防止correct_pred_score为空，还应该注意available_rec_list中的所有0元素，这里给的假数据要清零
                
                
                # 构造输入
                one_caption_prediction = {'pred_spans': correct_pred_rec.unsqueeze(0), # [bs, num_query, 2]
                                        'pred_logits': correct_pred_score.unsqueeze(0)} # [bs, num_query, 1]
                one_caption_target = timestamp_labels.unsqueeze(0) # 检查维度 [bs, num_query, 2]
                indices = self.matcher(one_caption_prediction, one_caption_target)
                matched_src_idx = self._get_src_permutation_idx(indices)
                matched_tgt_idx = self._get_tgt_permutation_idx(indices)
                
                ### rec_loss compute ###
                
                # label_loss
                src_logits = one_caption_prediction['pred_logits']
                target_label = torch.full(src_logits.shape[:-1], self.background_label, dtype=torch.int64, device=src_logits.device)
                target_label[matched_src_idx] = self.foreground_label
                one_label_loss = F.cross_entropy(src_logits.squeeze(0), target_label.squeeze(0), reduction="mean")
                
                # giou_loss and l1_loss
                tgt_spans = one_caption_target[matched_tgt_idx]
                src_spans = one_caption_prediction['pred_spans'][matched_src_idx]
                one_l1_loss = F.l1_loss(src_spans, tgt_spans, reduction="mean") # 归一化后数值比较小
                one_giou_loss = (1 - torch.diag(generalized_temporal_iou(src_spans, tgt_spans))).mean()
                
                
                iou_loss.append(one_giou_loss)
                error_loss.append(one_error_loss)
                l1_loss.append(one_l1_loss)
                label_loss.append(one_label_loss)
                cycle_loss.append(one_cycle_loss)

                # torch.distributed.barrier()
            
            iou_loss = torch.stack(iou_loss)
            l1_loss = torch.stack(l1_loss)
            error_loss = torch.stack(error_loss)
            label_loss = torch.stack(label_loss)
            cycle_loss = torch.stack(cycle_loss)
            
            error_loss = error_loss.sum()
            cycle_loss = cycle_loss.mean()
            
            available_mask = torch.tensor(available_list, dtype=torch.bool, device=rec_hidden_states.device)
            if available_mask.sum() == 0: # 无有效预测,防止除0，这里也不会死锁！！（因为不经过模型
                iou_loss = 0
                l1_loss = 0
                label_loss = 0
            else:
                iou_loss = iou_loss[available_mask].mean() # 平滑一下不同长度的损失
                l1_loss = l1_loss[available_mask].mean()
                label_loss = label_loss[available_mask].mean()
            
            rec_loss = iou_loss + error_loss + l1_loss*3 + label_loss + cycle_loss
            loss += rec_loss
                
                

        # deepspeed.comm.barrier() # 没用
        
        
        other_information = {
            # 'binary_loss': binary_loss.item() if type(binary_loss) == torch.Tensor else binary_loss,
            'lang_model_loss': lang_model_loss.item() if type(lang_model_loss) == torch.Tensor else lang_model_loss,
            'aug_lang_model_loss': aug_lang_model_loss.item() if type(aug_lang_model_loss) == torch.Tensor else aug_lang_model_loss,
            'iou_loss': iou_loss.item() if type(iou_loss) == torch.Tensor else iou_loss,
            'error_loss': error_loss.item() if type(error_loss) == torch.Tensor else error_loss,
            'l1_loss': l1_loss.item() if type(l1_loss) == torch.Tensor else l1_loss,
            'label_loss': label_loss.item() if type(label_loss) == torch.Tensor else label_loss,
            'cycle_loss': cycle_loss.item() if type(cycle_loss) == torch.Tensor else cycle_loss,
        }
        

        if not return_dict:
            output = (logits,) + outputs[1:] + (other_information,) # 检查一下
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            other_information=other_information, # 目前没用
        )
        
    def generate( # 用于评估
        self,
        input_ids: torch.LongTensor = None,
        im_src: Optional[torch.FloatTensor] = None,
        vi_src: Optional[torch.FloatTensor] = None,
        im_feat: Optional[torch.FloatTensor] = None,
        vi_feat: Optional[torch.FloatTensor] = None,
        timestamps: Optional[List[torch.FloatTensor]] = None,
        norm_timestamps: Optional[List[torch.FloatTensor]] = None,
        reverse_list: Optional[List[bool]] = None,
        captions: Optional[List[str]] = None,
        clip_length: Optional[float] = None,
        data_dict: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        *args, # 新加
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]: # 返回预测的string和rec解码的结果
        
        # # 动态修改类方法（chenwei nb！！# 换原生的lm_head,自己的太麻烦，在这里更新ids比较困难
        # self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)
        
        if inputs_embeds is None:
            (
                _,
                _,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal( # 检查这个函数是否可以使用
                input_ids, 
                None,  # position_ids
                attention_mask, 
                None,  # past_key_values
                None,  # label
                im_src, 
                vi_src,
                im_feat, 
                vi_feat, 
                norm_timestamps,
                reverse_list,
            )
        
        # outputs = super(LlavaLlamaModel, self).generate( # 不用这个因为这里内部也是调用llm的生成函数
        outputs = self.llm.generate( 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        ) # 加载错误参数很慢(zero3慢，大概一个样本运行了10min)不使用deepspeed能快一点（大概1-2min）（可能是因为不知道什么时候输出结束符）
        
        rec_ids = (outputs.sequences[:, :-1] == self.convert_tokens_to_ids(DEFAULT_MOMENT_TOKEN)).squeeze(0).nonzero().squeeze(-1) # 返回对应rec在文本序列的位置，nonzero返回非零元素的索引
        hidden_states = outputs.hidden_states # 所有层的隐藏层维度,[:, :-1]是为了去掉最后一个<|end_of_text|>
        # shape == (generated_token_num, layer_num, num_return_sequences*batch_size, seq_len, hidden_size)
        # 第一个token(1,416,4096) 第二个及之后的token(1,1,4096) 如果使用beam_search=2则输出的维度变为(2,1,4096)

        rec_hidden_states = [] # 从生成的文本中查找出对应的rec位置
        
        pred_recs = None
        pred_scores = None
        
        if hasattr(outputs, "beam_indices"): # beam size > 1
            for rid in rec_ids:
                beam_idx = outputs.beam_indices[0, rid] # 其中的shape为[bs, seq_len] # seq_len是生成的时间步
                rec_h = hidden_states[rid][-1][beam_idx]
                rec_hidden_states.append(rec_h.squeeze(0))
        else: # 贪心算法 beam size == 1
            for rid in rec_ids:
                rec_h = hidden_states[rid][-1]
                rec_hidden_states.append(rec_h.squeeze(0).squeeze(0))
        
        
        if len(rec_hidden_states) > 0:
            rec_hidden_states = torch.stack(rec_hidden_states).to(rec_h.dtype)
            pred_recs = self.rec_decoder(rec_hidden_states) # rec的st ed坐标
            pred_scores = F.softmax(self.rec_score_head(rec_hidden_states), dim=1)[:, self.foreground_label] # rec的得分

            # topk方案
            if len(pred_recs) < self.topk:
                topk_score_indices = torch.topk(pred_scores, len(pred_recs), dim=0).indices
            else:
                topk_score_indices = torch.topk(pred_scores, self.topk, dim=0).indices 
            
            pred_scores = pred_scores[topk_score_indices] # 这个不会报错
            
            # yuning's magic
            pred_recs_1= copy.deepcopy(pred_recs) # 这里一定要换变量名（猜测是内存空间有问题(就是不能用pred_recs这个变量名)
            pred_recs_1 = pred_recs_1[topk_score_indices, :] 
            
            
            
            # 阈值的方案
            # pred_recs = pred_recs[pred_scores > self.score_threshold] 
            # if len(pred_recs) == 0:
            #     pred_recs = pred_recs[torch.argmax(pred_scores)]
                
            # windows_and_scores = torch.cat((pred_recs, pred_scores.unsqueeze(-1)), dim=1) # 这有问题
            
            # pred_recs = pred_recs.clone().detach().requires_grad_(False)
            # pred_scores = pred_scores.clone().detach().requires_grad_(False)
            
            pred_recs_and_scores = torch.cat((pred_recs_1, pred_scores.unsqueeze(-1)), dim=1)
            pred_recs_and_scores = self.post_processor(pred_recs_and_scores, moment_length=clip_length) # 后处理(包含将归一化坐标转换为原始坐标)
            return outputs.sequences, pred_recs_and_scores.unsqueeze(0)
        
        else: # 这里没有预测到<rec>，直接返回(返回空数据会报错，返回忽略值)
            pred_recs_and_scores = -100*torch.ones(1, self.topk, 3, device=rec_ids.device, dtype=rec_ids.dtype)
            return outputs.sequences, pred_recs_and_scores 
    
        
    # def _update_model_kwargs_for_generation( # 该函数用于更新model_kwargs
    #     self,
    #     outputs,
    #     model_kwargs,
    #     is_encoder_decoder=False,
    #     standardize_cache_format=False,
    # ):
    #     model_kwargs = super(DtosForLocLM, self)._update_model_kwargs_for_generation(outputs,
    #                                                                             model_kwargs,
    #                                                                             is_encoder_decoder,
    #                                                                             standardize_cache_format)
    #     model_kwargs.update({"hidden_states": outputs.hidden_states})
    #     return model_kwargs
        
        
    def predict_rec( # 单次预测返回对应坐标位置,没改，可能没用
        self,
        **kwargs,
    ):
        pass
    


    