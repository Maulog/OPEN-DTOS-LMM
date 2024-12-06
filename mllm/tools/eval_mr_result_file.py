import torch
import sys
sys.path.append('/home/tianjirui/DTOS-LMM')
import os

from mllm.utils.box_ops import rec_iou_in_caption
from mllm.dataset.utils.io import save_to_json, save_to_jsonl

base_path = '/share_ssd/tianjirui/dtos_output/stage1/exp102/eval_top5/' # /checkpoint-70000
datasets_list = [
    'multitest_act_cap_test_prediction.pth',
    'multitest_charadessta_test_prediction.pth',
    'multitest_didemo_test_prediction.pth',
    # 'multitest_qv_test_prediction.pth', # 不用打开注释
    'multitest_qv_val_prediction.pth'
]
topk_list = [1,2,3,4,5]

def calculate_mr_metric(preds, dataset_name, topk = None):
    extract_failed = 0

    with torch.no_grad(): # 这里注意数据格式
        ious = []
        for i, pred in enumerate(preds):
            pred_rec_and_score = pred['pred_ts'][0]
            pred_rec, pred_score = pred_rec_and_score[..., :2], pred_rec_and_score[..., -1]
            target_rec = pred['label_ts'][0]
            
            if pred['pred_num'] == 0: # 一般不太会发生
                extract_failed += 1
                continue
            
            pred_rec = pred_rec[:pred['pred_num']] if topk is None else pred_rec[:topk]
            target_rec = target_rec[:pred['label_num']]
            
            iou = rec_iou_in_caption(torch.tensor(pred_rec), torch.tensor(target_rec)) # 此处原来有 * 1000，参考rec
            iou = iou if type(iou) == torch.Tensor else torch.tensor(iou)
            ious.append(iou) 
            
        ious = torch.stack(ious)
        
    # NOTE: please note iou only calculate for success target
    return {
        'R@0.3': 1.0 * (ious > 0.3).sum().item() / (len(preds)-extract_failed),
        'R@0.5': 1.0 * (ious > 0.5).sum().item() / (len(preds)-extract_failed),
        'R@0.7': 1.0 * (ious > 0.7).sum().item() / (len(preds)-extract_failed),
        'miou': ious.mean().item(),
        'dataset_name': dataset_name,
        'extract_failed': extract_failed,
    }

for topk in topk_list:
    save_list = []
    for dataset in datasets_list:
        preds = torch.load(os.path.join(base_path, dataset), map_location='cpu')
        dataset_name = dataset.split('.')[0]
        res = calculate_mr_metric(preds, dataset_name, topk)
        save_list.append(res)
    os.makedirs(os.path.join(base_path, f'top{topk}'), exist_ok=True)
    save_path = os.path.join(base_path, f'top{topk}', f'top{topk}_multi_prediction.jsonl')
    save_to_jsonl(save_list, save_path)