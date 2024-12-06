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

from llava.model.language_model.llava_llama import LlavaLlamaModel
from mllm.dataset.utils.transform import de_norm_box_xyxy, de_norm_point_xyxy
from mllm.models.detr.matcher import DetrHungarianMatcher
from mllm.models.dtos.dtos_base import DtosLmmForCausalLM, DtosConfig, PartialTrainableEmbedding, SpecialTokenClassifier
from mllm.models.dtos.dtos_loss import masked_loss, bbox_loss, CycleLoss, language_modeling_loss, no_special_token_language_modeling_loss
from mllm.models.dtos.seg_selected_method import *
from mllm.models.sam2.sam2_image_predictor import SAM2ImagePredictor
from mllm.config.constants import *

from mllm.models.dtos.dtos_base import DtosLmmForCausalLM, DtosConfig, PartialTrainableEmbedding
from mllm.models.dtos.dtos_loc import CategoryDimensionFc
from mllm.models.sam.modeling_sam import SamForLMSeg
from mllm.models.sam.sam_loss import SamLoss
from mllm.models.sam2.build_sam import build_sam2, build_sam2_video_predictor

# torch.autograd.set_detect_anomaly(True) # 用于检查定位反向传播时报的错


class DtosForSegLM(DtosLmmForCausalLM): 
    # 这里仍然继承dtos lmm，但使用不到其中有关定位的头部（也为了节省显存，但是加载stage1的lora）
    def __init__(self, config: DtosConfig, *args, **kwargs):
        super(DtosForSegLM, self).__init__(config, *args, **kwargs)
        
        self.box_decoder = CategoryDimensionFc( # 注意这是定位头
            self.config.hidden_size, 
            [
                self.config.hidden_size,
                self.config.hidden_size,
                # self.config.hidden_size*2,
            ], 
            4, 
            dropout_rate=0.5
        )
                            
        
        self.matcher = DetrHungarianMatcher(
            cost_class=self.config.cost_class, 
            cost_bbox=self.config.cost_bbox, 
            cost_giou=self.config.cost_giou, 
        )
        
        
        self.seg_score_head = nn.Linear(self.config.hidden_size, 2) # 这个prompt是否可信
                
        
        self.sam2_img_predictor = None
        self.sam2_vid_predictor = None
        self.sam_loss = SamLoss()
        
        self.score_threshold = 0.3
        self.confidence_label = 1    # 1 for confident, 0 for not confident
        self.global_step = 0
        self.background_label = 1
        self.foreground_label = 0
        
        self.nms_iou_threshold = 0.7
        self.overlap_iou_threshold = 0.6
        
        # self.box_decoder.apply(self.initialize_weights) # 这里待调试
        # self.seg_score_head.apply(self.initialize_weights)
        self.post_init()

    def build_seg_token_projector(self, new_token_num, vocab_size, hidden_dim, init_weight = None, from_pretrained = False):
        self.seg_token_projector = PartialTrainableEmbedding(
            new_token_num, vocab_size, hidden_dim, self.get_input_embeddings(), init_weight
        )
        return self.seg_token_projector
            
    def get_seg_token_projector(self):
        return self.seg_token_projector

    def average_and_reassign(self, pred_box_score):
        '''
        input: [n, 2, 2]
        output: [n*2, 2]
        '''
        if pred_box_score.numel() == 0:
            return pred_box_score.view(0, 2)
        
        new_pred_box_score = []
        for i in range(0, len(pred_box_score)):
            one_box_score = pred_box_score[i]
            N = one_box_score.shape[0]
            average_box = one_box_score.mean(dim=0, keepdim=True).repeat(N, 1)
            new_pred_box_score.append(average_box)
            
        new_pred_box_score = torch.stack(new_pred_box_score, dim=0).view(pred_box_score.shape[0]*2, 2)
        
        return new_pred_box_score
        
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
        im_src, im_feat, vi_feat,
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
                n_images = (input_ids[0] == self.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)).sum() # 图片的总数
                # 上一行仅适用于bs=1的
                image_features = self.encode_images_low_usage(im_src).to(self.device) # 并返回视频
                assert n_images == image_features.shape[0], 'The number of images token must be equal to the number of image features'
                # video_features, image_features = self.select_vision_feats(image_features, n_images) # 待灵活修改
            
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
        
        
        input_embeds = self.seg_token_projector(input_ids_copy) 
        

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

    def forward(
        self,
        im_src: Optional[torch.FloatTensor] = None,
        im_feat: Optional[torch.FloatTensor] = None,
        vi_feat: Optional[torch.FloatTensor] = None,
        orig_image_list: Optional[List] = None,
        orig_size: Optional[torch.FloatTensor] = None, # [h, w]
        frames_idx: Optional[torch.LongTensor] = None,
        captions: Optional[List[str]] = None,
        tgt_frm_idx: Optional[torch.LongTensor] = None,
        norm_bbox: Optional[torch.FloatTensor] = None, # 这是总的框不细致
        masks: Optional[torch.FloatTensor] = None, # [n_frms, h, w]
        valid: Optional[torch.FloatTensor] = None,
        # objs_masks_dict: Optional[dict] = None,
        tgt_norm_bbox_list: Optional[List[torch.FloatTensor]] = None,
        data_dict: Optional[dict] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]: # 这里仿照dtos_loc修改
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
                # vi_src, 
                im_feat, 
                vi_feat, 
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

        if self.global_step < 300:
            self.global_step += 1
            
        if labels is not None: 
            lang_model_loss = language_modeling_loss(logits.squeeze(0), new_labels.squeeze(0), len(self.tokenizer))
            # enhance the language without special tokens
            aug_lang_model_loss = no_special_token_language_modeling_loss(
                logits.squeeze(0), new_labels.squeeze(0), len(self.tokenizer), 
                self.convert_tokens_to_ids(DEFAULT_BOX_TOKEN)) # 这里用list传入多个
            
            loss += lang_model_loss + aug_lang_model_loss*3 # 这里暂时没加入信息统计
            
            # binary_loss = seg_binary_loss(mask_prob[answer_mask], new_labels[answer_mask], self.convert_tokens_to_ids(DEFAULT_SEG_TOKEN))
            # loss += binary_loss # 二分类没做移位！！！

        
        # seg_loss
        seg_loss = 0
        error_loss = []
        box_giou_loss = []
        box_l1_loss = []
        sam_loss = []
        label_loss = []
        if masks is not None and len(masks) > 0: # 有掩码的情况下
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
            box_token_mask = (all_pred_ids == self.convert_tokens_to_ids(DEFAULT_BOX_TOKEN)) & pred_mask # 这里筛选出预测部分的<box>
            all_pred_ids_without_ignore = all_pred_ids.clone().detach()
            all_pred_ids_without_ignore[~pred_mask] = IGNORE_INDEX
            
            # 此处应该小心，应仔细检查解码的ids(通常包含词前方的空格)！！！！并且此处应该忽略all_pred_ids中被掩码的部分
            start_delim = ' shows the following' # ' locate these moments' # 这里长的可能会检测不到，怀疑超过了数据的长度
            end_delim = ' the places depicted' # ' my detailed answers' # 这里待修改
            
            start_ids = self.tokenizer(start_delim)['input_ids']
            end_ids = self.tokenizer(end_delim)['input_ids']
            
            seg_st_token_idx = find_subseq_idx(all_pred_ids_without_ignore, start_ids) # 不统计前面忽略的部分
            seg_ed_token_idx = find_subseq_idx(all_pred_ids_without_ignore, end_ids)
            
            box_token_idx = torch.where(box_token_mask)[0]
            
            seg_st_token_num = len(seg_st_token_idx)
            seg_ed_token_num = len(seg_ed_token_idx)
            bbox_token_num = len(box_token_idx)
            
            box_id = self.convert_tokens_to_ids(DEFAULT_BOX_TOKEN)
            
            
            # available_list是语言建模中正确的预测（这个是回答中的）
            available_list = [0]*len(tgt_frm_idx) # 用来保存每个预测之间的<seg>数量，这里是由语言建模结果得到
            if seg_st_token_num == seg_ed_token_num == len(tgt_frm_idx) and seg_st_token_num and bbox_token_num:
                # and len(valid_token_idx) == len(tgt_frm_idx): # 有相同数量的<start>和<end>，且不为0
                # 过滤只取<seg_start>和<seg_end>之间的<seg>部分
                pred_target_list = [] # 此处用于更新available_list和seg_token_mask，防止出现<end><seg>这种情况
                st_ed_patten = torch.zeros_like(all_pred_ids, dtype=torch.bool, device=all_pred_ids.device)
                
                for i in range(len(tgt_frm_idx)): # 处理每个回答
                    st_idx, ed_idx = seg_st_token_idx[i], seg_ed_token_idx[i] # 计算seg_start和seg_end之间的<seg>数量
                    st_ed_patten[st_idx:ed_idx+1] = True # 用于只取开始结束中间的<seg>部分
                    sub_interval = all_pred_ids[st_idx:ed_idx+1]
                    
                    # 分别统计正负点的数量
                    cur_bbox_num = (sub_interval == box_id).sum().item()
                    
                    pred_target_list.append(cur_bbox_num) # 统计每个caption的<seg>数量
                    
                box_token_mask = box_token_mask & st_ed_patten # 只取<seg_start>和<seg_end>之间的<seg>部分
                available_list = pred_target_list
                
            box_hidden_states = last_hidden_states.squeeze(0)[box_token_mask] # 此处生成<seg>的hs
            
            #（这里不需要再次检查，因为如果出现推描述的时候推出了<seg>会造成卡死，只需要保证available_list是无误的就可以）

            offset_box = 0
            for i, cur_frm_idx in enumerate(tgt_frm_idx): # 遍历每一帧
                # get labels
                tgt_id = tgt_frm_idx[i] # label
                tgt_mask, tgt_valid = masks[tgt_id], valid[tgt_id]
                tgt_norm_bbox = tgt_norm_bbox_list[tgt_id] # 这里应该完善！！！判断是否为空tgt_valid（尝试添加到图片这里）！！！并进行调试！！！
                
                
                cur_bbox_num = available_list[i] # 此时available_list不再表示有效预测数
                
                # compute bbox
                cur_box_hs = box_hidden_states[offset_box:offset_box+cur_bbox_num]
                pred_box = self.box_decoder(cur_box_hs)
                pred_box = F.sigmoid(pred_box) # 限制了输出范围后，loss范围比较正常 shape=[bs, 2] e.g. [n,2]
                pred_box_score = self.seg_score_head(cur_box_hs) # [bs, 2]
                offset_box += cur_bbox_num
                
                
                # error_loss
                error_mask = (pred_box[:, 0] >= pred_box[:, 2]) | (pred_box[:, 1] >= pred_box[:, 3]) # 无效的预测的掩码
                error_pred_box = pred_box[error_mask]
                # 此损失用于惩罚无效的预测，用求和是防止分母为0，*10是为了放大错误的惩罚
                one_error_loss = torch.clamp((error_pred_box[:, 0] - error_pred_box[:, 2]), min=0).sum()*5 + \
                                torch.clamp((error_pred_box[:, 1] - error_pred_box[:, 3]), min=0).sum()*5
                
                correct_pred_box = pred_box[~error_mask]
                correct_pred_box_score = pred_box_score[~error_mask] # 只有box会有正确错误，点不会有
                if len(correct_pred_box) == 0: # 预测的box数量必须存在
                    available_list[i] = 0 # 不计算后续的box iou l1 loss
                    correct_pred_box = torch.tensor([[0,0,1,1]], dtype=correct_pred_box.dtype, device=correct_pred_box.device,
                                                   requires_grad=False).detach()
                    correct_pred_box_score = torch.tensor([[1,3]], dtype=correct_pred_box.dtype, device=correct_pred_box.device).detach()
                else:
                    pass # 没有拦住，还是无效预测
                
                
                if len(tgt_norm_bbox) == 0 or tgt_valid == 0: # 标签box必须存在，构造假标签
                    available_list[i] = 0 # 不计算后续的box iou l1 loss
                    tgt_norm_bbox = torch.tensor([[0,0,0.5,0.5]], dtype=correct_pred_box.dtype, device=correct_pred_box.device,
                                                   requires_grad=False).detach()
                else: # 拼接已有的目标
                    tgt_norm_bbox = torch.stack(tgt_norm_bbox, dim=0)
                    
                
                ### seg_loss compute ###
                
                # construct input
                one_frame_prediction = {'pred_boxes': correct_pred_box.unsqueeze(0), # [bs, num_query, 2]
                                        'pred_logits': correct_pred_box_score.unsqueeze(0)} # [bs, num_query, 1]
                one_frame_target = tgt_norm_bbox.unsqueeze(0) # 检查维度 [bs, num_query, 4]
                indices = self.matcher(one_frame_prediction, one_frame_target)
                matched_src_idx = self._get_src_permutation_idx(indices)
                matched_tgt_idx = self._get_tgt_permutation_idx(indices)
                
                # label_loss 
                src_logits = one_frame_prediction['pred_logits']
                target_label = torch.full(src_logits.shape[:-1], self.background_label, dtype=torch.int64, device=src_logits.device)
                target_label[matched_src_idx] = self.foreground_label # 这里是matched_src_idx？
                one_label_loss = F.cross_entropy(src_logits.squeeze(0), target_label.squeeze(0), reduction="sum")
                
                # box_giou_loss and box_l1_loss
                tgt_boxes = one_frame_target[matched_tgt_idx]
                src_boxes = one_frame_prediction['pred_boxes'][matched_src_idx]
                one_box_l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="mean") # 归一化后数值比较小
                one_box_giou_loss = 1 - generalized_box_iou(src_boxes, tgt_boxes).diagonal().mean()
                
                    
                # sam_loss(supervision point)
                # sam2 image prediction
                tgt_frame = orig_image_list[tgt_id]
                self.sam2_img_predictor.set_image(tgt_frame)
                
                
                if available_list[i] == 0: # 不存在有效预测，或者不存在有效box，构造假orig_size_correct_box
                    orig_size_correct_box = torch.tensor([[0,0,2,2]], dtype=torch.int32)
                else: # 存在有效预测
                    orig_size_correct_box = de_norm_box_xyxy(correct_pred_box, w=orig_size[1], h=orig_size[0])
                
                pred_masks, pred_mask_scores, _ = self.sam2_img_predictor.predict_in_training(
                    point_coords=None,   # 原有尺度
                    point_labels=None,
                    box=orig_size_correct_box,
                    multimask_output=False,
                    return_logits=True,
                )
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                one_sam_loss = self.sam_loss(pred_masks, tgt_mask, pred_mask_scores, device=self.device)
                
                avail_bbox = bool(available_list[i])
                
                # 此处考虑了如果没有obj怎么计算loss（原计划根据预测的分数进行筛选或者tgt_valid进行筛选，但感觉丢失信息）
                # 这里分box和point进行讨论，这里经过考虑后决定都不乘上tgt_valid
                # point在训练过程中只要有一定会经过sam生成mask，利用sam_loss去监督用pos_point在更小面积的地方而不是随机生成
                # label_loss独立计算代表了可信度，在训练阶段不会使用，但后期推理阶段可根据置信度分数进行过滤
                # 针对错误的或者没有bbox的情况，则已经在avail_bbox中进行剔除
                one_error_loss = one_error_loss
                one_box_giou_loss = one_box_giou_loss*avail_bbox
                one_box_l1_loss = one_box_l1_loss*avail_bbox
                one_label_loss = one_label_loss*avail_bbox
                one_sam_loss = one_sam_loss*avail_bbox
                
                # handle nan and []
                def check_nan_and_none(x):
                    if torch.isnan(x).sum() > 0 or x.numel() == 0:
                        return torch.tensor(0, dtype=self.dtype, device=self.device)
                    return x
                one_error_loss = check_nan_and_none(one_error_loss)
                one_box_giou_loss = check_nan_and_none(one_box_giou_loss)
                one_box_l1_loss = check_nan_and_none(one_box_l1_loss)
                one_label_loss = check_nan_and_none(one_label_loss)
                one_sam_loss = check_nan_and_none(one_sam_loss)
                
                error_loss.append(one_error_loss)
                box_giou_loss.append(one_box_giou_loss)
                box_l1_loss.append(one_box_l1_loss)
                label_loss.append(one_label_loss)
                sam_loss.append(one_sam_loss)

            
            error_loss = torch.stack(error_loss).sum()
            box_giou_loss = torch.stack(box_giou_loss).mean()
            box_l1_loss = torch.stack(box_l1_loss).mean()
            label_loss = torch.stack(label_loss).mean()
            sam_loss = torch.stack(sam_loss).mean()
            
            # debug用
            # label_loss = 0 # 不会进行反向传播，就是该损失影响反向传播
            # sam_loss = 0
            
            seg_loss = error_loss + (box_giou_loss + box_l1_loss*3)*3 + label_loss + sam_loss*0.5
            loss = loss + seg_loss
            
        
        other_information = {
            'lang_model_loss': lang_model_loss.item() if type(lang_model_loss) == torch.Tensor else lang_model_loss,
            'aug_lang_model_loss': aug_lang_model_loss.item() if type(aug_lang_model_loss) == torch.Tensor else aug_lang_model_loss,
            'box_giou_loss': box_giou_loss.item() if type(box_giou_loss) == torch.Tensor else box_giou_loss,
            'box_l1_loss': box_l1_loss.item() if type(box_l1_loss) == torch.Tensor else box_l1_loss,
            'error_loss': error_loss.item() if type(error_loss) == torch.Tensor else error_loss,
            'label_loss': label_loss.item() if type(label_loss) == torch.Tensor else label_loss,
            'sam_loss': sam_loss.item() if type(sam_loss) == torch.Tensor else sam_loss,
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
        # vi_src: Optional[torch.FloatTensor] = None,
        im_feat: Optional[torch.FloatTensor] = None,
        vi_feat: Optional[torch.FloatTensor] = None,
        orig_image_list: Optional[List[torch.FloatTensor]] = None,
        orig_size: Optional[List[torch.FloatTensor]] = None,
        frames_idx: Optional[List[bool]] = None,
        captions: Optional[List[str]] = None,
        tgt_frm_idx: Optional[float] = None,
        norm_bbox: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        valid: Optional[torch.FloatTensor] = None,
        # objs_masks_dict: Optional[dict] = None,
        tgt_norm_bbox_list: Optional[List[torch.FloatTensor]] = None,
        video_dir: Optional[str] = None,
        data_dict: Optional[dict] = None,
        
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        *args, # 新加
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]: # 返回预测的string和seg解码的结果
        
        self.sam2_vid_predictor.to(dtype=torch.float32) # 这里暂时不考虑判断是否是之前的视频，因为用多线程读取不会阻塞
        inference_state = self.sam2_vid_predictor.init_state(video_path=video_dir, async_loading_frames=True)
        self.sam2_vid_predictor.reset_state(inference_state)
        
        # # 动态修改类方法（chenwei nb！！# 换原生的lm_head,自己的太麻烦，在这里更新ids比较困难
        # self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)
        
        
        ############################循环开始################################
        # (优先！！)可以直接从hiddenstate中取出，然后直接拼接，但需要搞清楚attention_mask的作用
        
        history_emb = None
        ret = {}
        ret['tgt_frm_res'] = {}
        check_none_list = []
        all_frames_boxes_list = [] # [[[1,2,3,4], [1,2,3,4]] ...]
        all_frames_scores_list = []
        
        tgt_i_token_list = [f'<tgt_{i+1}>' for i in range(10)]
        tgt_i_id_list = [self.convert_tokens_to_ids(tgt_i) for tgt_i in tgt_i_token_list]
        tgt_i_map = {tgt_i:self.convert_tokens_to_ids(tgt_i) for tgt_i in tgt_i_token_list}
        tgt_i_map_reverse = {v: k for k, v in tgt_i_map.items()}
        
        for idx, tgt_id in enumerate(tgt_frm_idx): # 遍历每一帧
            # 拼接上次的结果（仅限于beam=1，和当前这个任务）
            tgt_frame_idx = frames_idx[tgt_id].item()
            tgt_norm_box = norm_bbox[tgt_id]
            tgt_mask = masks[tgt_id]
            
            for tgt_i_id in tgt_i_id_list: # 遍历每一个<tgt_i>token对应的id，如果有则替换
                indices = torch.nonzero(input_ids == tgt_i_id, as_tuple=False)
                if indices.shape[0] == 3: # 提问的目标帧，因为按现在的模板只有提问的地方会多一个
                    query_tgt_id = indices[-1]
                    input_ids[query_tgt_id[0], query_tgt_id[1]] = tgt_i_map[f'<tgt_{tgt_id+1}>'] # 修改对应提问的位置
                    break
            
            (
                _,
                _,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal( # 每次都重新拼接，替换提问的信息
                input_ids, 
                None,  # position_ids
                attention_mask, 
                None,  # past_key_values
                None,  # label
                im_src, 
                im_feat, 
                vi_feat, 
            )
            h, w = orig_size[0], orig_size[1]
            
            outputs = self.llm.generate( 
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, # 这里att和input_emb维度相同
                **kwargs,
            ) # 加载错误参数很慢(zero3慢，大概一个样本运行了10min)不使用deepspeed能快一点（大概1-2min）（可能是因为不知道什么时候输出结束符）
            
            pred_text = self.decode_generate_ids(outputs.sequences)
            all_pred_ids = outputs.sequences[:, :-1].squeeze(0) # 去掉最后一个<|end_of_text|>,begin对应之前的信息
            hidden_states = outputs.hidden_states # 所有层的隐藏层维度,[:, 1:]是为了去掉第一个<|begin_of_text|>
            # shape == (generated_token_num, layer_num, num_return_sequences*batch_size, seq_len, hidden_size)
            # 第一个token(1,416,4096) 第二个及之后的token(1,1,4096) 如果使用beam_search=2则输出的维度变为(2,1,4096)

            def get_seg_hidden_states(seg_ids, hidden_states, outputs):
                '''
                seg_ids: seg在文本序列的位置索引
                '''
                seg_hidden_states = [] # 从生成的文本中查找出对应的seg位置
                if hasattr(outputs, "beam_indices"): # beam size > 1
                    for sid in seg_ids:
                        beam_idx = outputs.beam_indices[0, sid] # 其中的shape为[bs, seq_len] # seq_len是生成的时间步
                        seg_h = hidden_states[sid][-1][beam_idx]
                        seg_hidden_states.append(seg_h.squeeze(0))
                else: # 贪心算法 beam size == 1
                    for sid in seg_ids:
                        seg_h = hidden_states[sid][-1]
                        seg_hidden_states.append(seg_h.squeeze(0).squeeze(0))
                        
                if seg_hidden_states == []: # 处理这里为空的情况
                    seg_hidden_states = torch.zeros([0,4096], dtype=hidden_states[0][0].dtype, device=self.device)
                else:
                    seg_hidden_states = torch.stack(seg_hidden_states).to(seg_h.dtype) # 这里单个可能要写单个token的情况
                return seg_hidden_states
            
            # 返回对应seg在文本序列的位置，nonzero返回非零元素的索引
            box_ids = (outputs.sequences[:, :-1] == self.convert_tokens_to_ids(DEFAULT_BOX_TOKEN)).squeeze(0).nonzero().squeeze(-1)
            box_hs = get_seg_hidden_states(box_ids, hidden_states, outputs)
            
            
            if len(box_hs): # 有预测的情况下
                pred_boxes = self.box_decoder(box_hs) # box的坐标
                pred_boxes = F.sigmoid(pred_boxes)
                pred_boxes_scores = F.softmax(self.seg_score_head(box_hs), dim=1)[:, self.foreground_label] # box的得分
                
                # post-check []，检查是否有box的预测
                filter_box = pred_boxes if pred_boxes.numel() > 0 else None # 检查当前预测是否为空
                check_none = [filter_box] # 添加list统计每个的检查结果
                is_all_none = all([x is None for x in check_none])
                check_none_list.append(is_all_none)

                # 这里如果直接返回mask容易造成显存不够和gpu、cpu通信问题，所以这里只传递llm的结果！！
                # 单独保存每一帧的结果
                all_frames_boxes_list.append(pred_boxes.to(dtype=torch.float32)) # 这里待调试
                all_frames_scores_list.append(pred_boxes_scores.to(dtype=torch.float32))
                
                # save dtos prediction  这里直接返回预测的prompts (这里应该控制none，后面无法进行拼接，包括下面的函数)
                ret['tgt_frm_res'][tgt_frame_idx] = {} # 这里之后再检查一次，看下保存的是否全
                ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes'] = pred_boxes
                ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes_scores'] = pred_boxes_scores
                
                # point没有预测
                ret['tgt_frm_res'][tgt_frame_idx]['pred_points'] = torch.zeros((0,2), dtype=self.dtype, device=self.device)
                ret['tgt_frm_res'][tgt_frame_idx]['pred_points_scores'] = torch.zeros((0), dtype=self.dtype, device=self.device)
                ret['tgt_frm_res'][tgt_frame_idx]['pred_pos_num'] = torch.tensor(0, device=self.device)
                
            else:
                check_none_list.append(True) # 为none则是true
                
                all_frames_boxes_list.append(None) # 这里待调试
                all_frames_scores_list.append(None)
                
                ret['tgt_frm_res'][tgt_frame_idx] = {}
                ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes'] = torch.zeros((0,4), dtype=self.dtype, device=self.device)
                ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes_scores'] = torch.zeros((0), dtype=self.dtype, device=self.device)
                
                ret['tgt_frm_res'][tgt_frame_idx]['pred_points'] = torch.zeros((0,2), dtype=self.dtype, device=self.device)
                ret['tgt_frm_res'][tgt_frame_idx]['pred_points_scores'] = torch.zeros((0), dtype=self.dtype, device=self.device)
                ret['tgt_frm_res'][tgt_frame_idx]['pred_pos_num'] = torch.tensor(0, device=self.device)
            
            
        ###################################################################################
        # 循环外，在所有预测帧中筛选出最优帧
        assert len(all_frames_boxes_list) == len(all_frames_scores_list) == len(tgt_frm_idx), 'all_frames_result should be recorded per frame'
        
        # 如果全为none则返回失败
        if all(check_none_list): # 如果全为none（全为true）则直接返回(当前所有tgt_frm没有有效预测)
            ret = self.return_failed(tgt_frm_idx, frames_idx)
            return outputs.sequences, ret
        
        # 过滤none
        tgt_frm_idx = [x for i, x in enumerate(tgt_frm_idx) if all_frames_boxes_list[i] is not None]
        all_frames_boxes_list = [x for x in all_frames_boxes_list if x is not None]
        all_frames_scores_list = [x for x in all_frames_scores_list if x is not None]
        
        ## ablation strategy box num
        box_num_limit = 1
        tmp_all_frames_boxes_list = []
        tmp_all_frames_scores_list = []
        for tgt_idx, one_frm_boxes, one_frm_scores in zip(tgt_frm_idx, all_frames_boxes_list, all_frames_scores_list):
            # 遍历每一帧的内容，然后根据每一帧中分数最高的留下框数
            if len(one_frm_boxes) > box_num_limit:
                scores, indices = one_frm_scores.topk(box_num_limit, largest=True)
                tmp_all_frames_boxes_list.append(one_frm_boxes[indices])
                tmp_all_frames_scores_list.append(scores)
            else:
                tmp_all_frames_boxes_list.append(one_frm_boxes)
                tmp_all_frames_scores_list.append(one_frm_scores)
                
        all_frames_boxes_list = tmp_all_frames_boxes_list
        all_frames_scores_list = tmp_all_frames_scores_list
        ##
        
        # 这里是筛选的策略
        filter_boxes, filter_scores, selected_tgt_id = method3(  # 这里设置不同的筛选方法 # 默认 method3
            all_frames_boxes_list, all_frames_scores_list, tgt_frm_idx, # 输入
            self.overlap_iou_threshold, self.nms_iou_threshold, # 超参
        )
        
        
        ###### 这里改为多帧预测的筛选方法 ######
        tgt_frame_idx = [frames_idx[indx].item() for indx in selected_tgt_id]
        ret['choosed_tgt_frm'] = tgt_frame_idx
        ret['filter_boxes'] = filter_boxes
        ret['filter_scores'] = filter_scores
        orig_filter_boxes = [de_norm_box_xyxy(filter_box, w=w, h=h) for filter_box in filter_boxes]
        
        obj_id = 0
        for tgt_idx, one_frm_orig_boxes in zip(tgt_frame_idx, orig_filter_boxes):
            for _, orig_filter_box in enumerate(one_frm_orig_boxes):
                _, out_obj_ids, out_mask_logits = self.sam2_vid_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=tgt_idx, # 对应的帧id
                    obj_id=obj_id,
                    points=None,
                    labels=None, # 其中label=2，3是box的左上和右下点
                    box=orig_filter_box,
                    device=self.device,
                )
                obj_id += 1
                if (obj_id > 3 and data_dict['clip'][1] > 180) or obj_id > 4: # 实际最多只能添加2个，用于控制显存
                    break
        
        # 这里是传播到整个视频的代码
        vid_mid_len = data_dict['clip'][1] / 2
        closest_value = min(tgt_frame_idx, key=lambda x: abs(x - vid_mid_len))
        anchor_idx = tgt_frame_idx.index(closest_value) # index只会返回第一个
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # 这里用上下文管理器防止类型不同报错
            # video_segments contains the per-frame segmentation results
            ret['pred_masks'] = {}
            
            # 首先正向传播
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_vid_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=anchor_idx,
                reverse=False): # False 表示正向传播，True 表示反向传播
                ret['pred_masks'][out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # 然后反向传播
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_vid_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=anchor_idx,
                reverse=True):
                ret['pred_masks'][out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        
        
        
        torch.cuda.empty_cache()
        
        return outputs.sequences, ret
            
    def return_failed(self, tgt_frm_idx, frame_idx): # 待修改
        ret = {}
        ret['tgt_frm_res'] = {}
        ret['choosed_tgt_frm'] = None
        ret['filter_boxes'] = None
        ret['filter_scores'] = None
        for i, tgt_idx in enumerate(tgt_frm_idx):
            tgt_frame_idx = frame_idx[tgt_idx].item()
            ret['tgt_frm_res'][tgt_frame_idx] = {}
            ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes'] = torch.zeros((0,4), dtype=self.dtype, device=self.device)
            ret['tgt_frm_res'][tgt_frame_idx]['pred_boxes_scores'] = torch.zeros((0), dtype=self.dtype, device=self.device)
            
            ret['tgt_frm_res'][tgt_frame_idx]['pred_points'] = torch.zeros((0,2), dtype=self.dtype, device=self.device)
            ret['tgt_frm_res'][tgt_frame_idx]['pred_points_scores'] = torch.zeros((0), dtype=self.dtype, device=self.device)
            ret['tgt_frm_res'][tgt_frame_idx]['pred_pos_num'] = torch.tensor(0, device=self.device)
            
        ret['pred_masks'] = None
        torch.cuda.empty_cache()
        return ret
            