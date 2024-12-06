import json
import logging
import os
import sys
import re
from typing import Optional, List, Union, Tuple
import warnings

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
from mllm.models.dtos.dtos_base import DtosLmmForCausalLM, DtosConfig, PartialTrainableEmbedding, SpecialTokenClassifier
from mllm.models.dtos.dtos_loc import DtosForLocLM
from mllm.models.dtos.dtos_loss import masked_loss, bbox_loss, CycleLoss, language_modeling_loss, compute_rec_loss, rec_binary_loss
from mllm.utils.box_ops import box_iou
from mllm.config.constants import *


    
    

class DtosForDebug(DtosForLocLM):
    def __init__(self, config: DtosConfig, *args, **kwargs):
        super(DtosForDebug, self).__init__(config, *args, **kwargs) # 直接使用encode_input     
        self.post_init()
        
    
    def forward(self,
        input_ids: torch.LongTensor = None,
        im_src: Optional[torch.FloatTensor] = None,
        # vi_src: Optional[torch.FloatTensor] = None,
        im_feat: Optional[torch.FloatTensor] = None,
        vi_feat: Optional[torch.FloatTensor] = None,
        timestamps: Optional[List[torch.FloatTensor]] = None,
        norm_timestamps: Optional[List[torch.FloatTensor]] = None,
        captions: Optional[List[str]] = None,
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
        
        return outputs
        
    