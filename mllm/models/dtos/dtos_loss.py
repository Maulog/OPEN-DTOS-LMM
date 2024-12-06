import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist

from mllm.utils.box_ops import generalized_box_iou, rec_iou_in_caption
from mllm.config.constants import *

def language_modeling_loss(logits, labels, vocab_size): # 移除最后一个时间步预测和当前实际词概率，并将其重塑为适合计算交叉熵损失的形式
    # [:-1] 移除最后一个时间步
    shift_logits = logits[:-1][..., :-1, :].contiguous() # 只能看到之前的位置
    shift_labels = labels[:-1][..., 1:].contiguous() # 不预测第一个标签
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model/pipeline parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def no_special_token_language_modeling_loss(logits, labels, vocab_size, special_token_id):
    if type(special_token_id) == list:
        special_token_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in special_token_id:
            special_token_mask |= labels == token_id
        new_labels = labels.clone()
        new_labels[special_token_mask] = IGNORE_INDEX
    else:
        special_token_mask = labels == special_token_id
        new_labels = labels.clone() # 避免就地操作
        new_labels[special_token_mask] = IGNORE_INDEX
    
    loss = language_modeling_loss(logits, new_labels, vocab_size)
    return loss

def rec_binary_loss(binary_pred, binary_label, special_token_id):
    shift_binary_pred = binary_pred[:-1] # 移除最后一个时间步
    shift_binary_label = binary_label[1:]
    
    shift_binary_label = (shift_binary_label==special_token_id).to(shift_binary_pred.dtype) # 构造onehot编码作为标签
    sample_weight = (shift_binary_label==1).to(shift_binary_pred.dtype) + 1 # 给正例添加权重，防止负例过多
    return F.binary_cross_entropy_with_logits(shift_binary_pred, shift_binary_label, weight=sample_weight)


class CycleLoss(nn.Module):
    def __init__(self, loc_encoder, loc_decoder):
        super(CycleLoss, self).__init__()
        self.loc_encoder = loc_encoder
        self.loc_decoder = loc_decoder

    def forward(self, pred_locs, selected_hidden_states, loc_embeds, loc_inputs):
        pred_output_embeds = self.loc_encoder(pred_locs) # 计算输入的位置编码后产生的嵌入与原本的嵌入的差异
        cycle_loss1 = F.mse_loss(pred_output_embeds, selected_hidden_states, reduction="none")

        pred_input_locs = self.loc_decoder(loc_embeds) # 计算输入的嵌入还原后产生的位置编码与原本的位置编码的差异
        cycle_loss2 = F.l1_loss(pred_input_locs, loc_inputs, reduction="none")
        
        cycle_loss1 = cycle_loss1.mean() if len(cycle_loss1)!=0 else 0
        cycle_loss2 = cycle_loss2.mean() if len(cycle_loss2)!=0 else 0
        
        return cycle_loss1 + cycle_loss2

def is_empty_tensor(tensor):
    return tensor.numel() == 0


def compute_rec_loss(pred_rec, target_rec, indices): # 计算logit与label的损失，计算iou和l1的损失
    # 一个caption的rec
    

    
    ''' # 这里重写，用momentdetr的代码
    l1_loss = 0
    if correct_pred_rec.shape[0] == 1 and target_rec.shape[0] == 1:
        l1_loss = F.l1_loss(correct_pred_rec[0], target_rec[0], reduction='mean')
    
    iou_loss = 1
    if correct_pred_rec.shape[0] != 0:
        iou_loss = 1 - rec_iou_in_caption(correct_pred_rec, target_rec)
    '''
    
    
    return one_label_loss, one_iou_loss, one_l1_loss


def bbox_loss(src_boxes, target_boxes):
    l1_loss = F.l1_loss(src_boxes, target_boxes, reduction='none').mean()
    # l1_loss = masked_loss(l1_loss, 1)

    mask = (src_boxes[:, 2:] > src_boxes[:, :2]).all(-1) # 用于筛选出有效的候选框（右下角>左上角）
    src_boxes = src_boxes[mask]
    target_boxes = target_boxes[mask]
    # if not mask.all():
    #     print(len(mask)-mask.sum())

    loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
    loss_giou = loss_giou.mean()
    # loss_giou = masked_loss(loss_giou, 1) # 最后一个元素mask
    return l1_loss*2 + loss_giou/5  # 此处设置返回权重

def masked_loss(loss, n): # 将后n个元素mask住
    mask = torch.ones_like(loss)
    mask[-n:] = 1e-10
    loss = (loss * mask).sum()/(mask.sum()) # 平均，只计算没有被mask的元素
    return loss