import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import transformers
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from mllm.utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor, \
    CLIPVisionConfig, BitsAndBytesConfig
from llava.model.language_model.llava_llama import LlavaLlamaModel
from llava.model.language_model.llava_llama import LlavaLlamaConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from mllm.config.constants import *

import os, sys, os.path as osp
import warnings
from abc import ABC, abstractmethod

import torch, logging

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)



from llava.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    vision_resolution_elevation,
    unit_test_rope_scaling,
)

from collections import OrderedDict
from llava.model.utils import get_model_config
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.configuration_llava import LlavaConfig

from transformers.modeling_utils import ContextManagers, no_init_weights



class DtosConfig(AutoConfig): # 这里可能要改
    model_type = "dtos" # FIXME
    rec_dim = None
    seg_dim = None
    sam_path = None
    sam2_cfg = None
    sam2_path = None
    device = None
    


class PartialTrainableEmbedding(nn.Module): # 仅输入embedding
    def __init__(self, new_token_num: int, vocab_size: int, hidden_dim: int,
                 orig_emb: nn.Embedding, init_weight = None) -> None:
        super(PartialTrainableEmbedding, self).__init__()
        self.orig_emb = orig_emb
        self.emb = nn.Parameter(torch.randn(new_token_num, hidden_dim))
        # self.emb = nn.Embedding(new_token_num, hidden_dim)
        if init_weight is not None:
            assert self.emb.data.shape == init_weight.shape, "init_weight shape not match" 
            self.emb.data = init_weight
        self.vocab_size = vocab_size
        self.new_token_num = new_token_num
        self.orig_vocab_size = vocab_size - new_token_num
        self.hidden_dim = hidden_dim
        
        self.emb.requires_grad = False
        self.orig_emb.requires_grad = False
        
    def init_train_requires_grad(self) -> None:
        self.emb.requires_grad = True
        self.orig_emb.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # 整个seq, 需要考虑加入batch_size维度
        x = x.long()
        x_emb = torch.zeros((x.shape[0], x.shape[1], self.hidden_dim),
                            dtype=self.emb.dtype, device=self.emb.device)
        for i in range(x.shape[0]):
            x_i = x[i]
            mask = x_i.ge(self.orig_vocab_size) # >= 给新token加mask
            x1, x2 = x_i[mask], x_i[~mask]
            # x1, x2 = torch.masked_select(x, mask), torch.masked_select(x, ~mask)
            
            x1 = x1 - self.orig_vocab_size
            x2_emb = self.orig_emb(x2)
            x1_emb = self.emb[x1] # 新增的token
            
            # 合并
            x_emb[i][mask] = x1_emb
            x_emb[i][~mask] = x2_emb
        return x_emb
    
    def merge_to_embedding(self) -> None: # 衔接保存操作
        self.orig_emb.weight.data[-self.new_token_num:] = self.emb.data
        
        
class SpecialTokenClassifier(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super(SpecialTokenClassifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1, bias=False)
        self.threshold = 0.5 # nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        
        self.fc.requires_grad = False
        
    def forward(self, x: torch.Tensor, label = None) -> torch.Tensor: # 返回概率
        prob = torch.sigmoid(self.fc(x)).squeeze(-1)
        mask = prob > self.threshold
        return mask, prob
    
    def init_train_requires_grad(self) -> None:
        self.fc.requires_grad = True
    
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x, y)
    

        
    
    



class DtosLmmForCausalLM(LlavaLlamaModel):
    config_class = DtosConfig
    def __init__(self, config: DtosConfig, *args, **kwargs) -> None:
        super(DtosLmmForCausalLM, self).__init__(config, *args, **kwargs)
        
    @classmethod
    def load_visiontower_and_projection(self, config: PreTrainedModel = None, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower")  or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        
        device = kwargs["device_map"]
        self.vision_tower.to(device=device)
        self.mm_projector.to(device=device)
        
        return self.vision_tower, self.mm_projector

        # if getattr(config, "vision_tower_cfg", None) is None:
        #     self.config.vision_tower_cfg = self.vision_tower.config
        # if getattr(self.config, "mm_projector_cfg", None) is None:
        #     self.config.mm_projector_cfg = self.mm_projector.config
            
    
    def convert_tokens_to_ids(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def decode_generate_ids(self, ids: torch.Tensor) -> Union[List[str], str]:
        assert ids.ndim in [1, 2]
        only_one_sentence = ids.ndim == 1
        if only_one_sentence:
            ids = ids.unsqueeze(0)
        ids = copy.deepcopy(ids)  # do not modify origin preds and targets
        ids[ids < 0] = self.tokenizer.pad_token_id
        res = self.tokenizer.batch_decode(ids, clean_up_tokenization_spaces=True)
        if only_one_sentence:
            return res[0]
        return res
        
        
    
    def encode_images_low_usage(self, images: torch.Tensor, bs = 16) -> torch.Tensor:
        num_images = images.shape[0]
        batches = [(i * bs, min((i + 1) * bs, num_images)) for i in range((num_images + bs - 1) // bs)]
        img_feats = []

        for index, (start, end) in enumerate(batches):
            images_part = images[start:end]
            image_features = self.encode_images(images_part)
            img_feats.append(image_features)
            
        return torch.cat(img_feats, dim=0)
        
    
    def select_vision_feats(self, img_feats: torch.Tensor, selected_num): # TODO: 此处尝试其他的方法选择融合特征
        selected_indices = np.linspace(0, 99, selected_num, endpoint=False, dtype=int) # [9, 29, 49, 69, 89] # 待修改为自动判断长度的
        video_feats = img_feats.mean(dim=1) # 选取视频的cls特征
        image_feats = img_feats[selected_indices, :, :]
        return video_feats, image_feats

    
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
