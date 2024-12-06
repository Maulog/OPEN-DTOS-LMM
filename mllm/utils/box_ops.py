# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def merge_overlapping_intervals(timestamps): # 并集
    timestamps = timestamps[timestamps[:, 0].sort().indices]
    merged = None
    
    for timestamp in timestamps:
        timestamp = timestamp.unsqueeze(0)
        if merged is None:
            merged = timestamp
        elif timestamp[0][0] > merged[-1][1]:
            merged = torch.cat((merged, timestamp))
        else:
            # merged[-1][1] = max(merged[-1][1], timestamp[0][1]) # 有问题
            updated_last_element = torch.cat(
                (merged[:-1], torch.tensor(
                    [[merged[-1][0], max(merged[-1][1], timestamp[0][1])]], device=merged.device, dtype=merged.dtype
                )), dim=0
            )
            merged = updated_last_element  
            # 这里仍然是就地操作
    return merged

def find_all_intersections(pred, target): # 交集
    intersections = None
    for interval1 in pred:
        for interval2 in target:
            start = max(interval1[0], interval2[0]).unsqueeze(0)
            end = min(interval1[1], interval2[1]).unsqueeze(0)
            if start <= end: 
                avail = torch.cat((start, end), dim=0).unsqueeze(0)
                if intersections is None:
                    intersections = avail
                else:
                    intersections = torch.cat((intersections, avail), dim=0)
                    
    return intersections

def rec_iou_in_caption(pred_recs, target_recs, scaling=100): # 该函数不能用于模型训练
    '''
        针对一个caption下的生成rec和目标rec的iou，
        输入的数量可以不相等，src_rec.shape==[n,2], target_rec.shape==[m,2]
        先分别取src_rec, target_rec各自的并集，
        再计算二者的交集        
    '''
    # 这是有问题的，没有通过gradcheck的测试
    src_rec_union = merge_overlapping_intervals(pred_recs)
    target_rec_union = merge_overlapping_intervals(target_recs)
    
    unions = merge_overlapping_intervals(torch.cat((src_rec_union, target_rec_union), dim=0))
    intersections = find_all_intersections(src_rec_union, target_rec_union)
    if intersections is None: # 没有并集
        return 0
    
    unions_value = torch.sum(unions[:, 1] - unions[:, 0])
    intersections_value = torch.sum(intersections[:, 1] - intersections[:, 0])
    
    iou = intersections_value / unions_value if unions_value != 0 else 0
    
    # 离散方法，但可能在反向的时候不会有梯度
    # def discrete(src, scaling=scaling):
    #     discrete_src = torch.zeros(scaling, device=src.device, dtype=torch.int16)
    #     src = torch.clamp(torch.round(src*scaling), min=0, max=scaling).int()
    #     for i in range(src.shape[0]):
    #         src_rec = src[i]
    #         discrete_src[src_rec[0]:src_rec[1]].fill_(1) 
    #     return discrete_src
    
    # pred_recs = torch.clamp(pred_recs, min=0, max=1)
    # target_recs = torch.clamp(target_recs, min=0, max=1)
    
    # # tensor version  使用离散的方法最后进行计数
    # discrete_pred_recs = discrete(pred_recs, scaling)
    # discrete_target_recs = discrete(target_recs, scaling)
    
    # union = ((discrete_pred_recs + discrete_target_recs) > 0).sum()
    # intersections = ((discrete_pred_recs + discrete_target_recs) == 2).sum()
    
    # iou = intersections / union if union != 0 else 0
    
    return iou
    
    

def rec_iou(src_rec, target_rec): # 该函数需要src_rec和target_rec维度相等 for [st,end], shape==[bs,2]
    inner_st = torch.max(src_rec[:, 0], target_rec[:, 0]) 
    inner_end = torch.min(src_rec[:, 1], target_rec[:, 1]) 
    
    outer_st = torch.min(src_rec[:, 0], target_rec[:, 0])
    outer_end = torch.max(src_rec[:, 1], target_rec[:, 1])
    
    union = outer_end - outer_st
    inner = (inner_end - inner_st).clamp(min=0)
    
    iou = inner / union if union != 0 else 0
    return iou

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


from packaging import version

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)