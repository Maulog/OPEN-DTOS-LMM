import numpy as np
import torch
import os
import sys
sys.path.append('/home/tianjirui/DTOS-LMM')

from mllm.dataset.single_video_dataset.mr import pre_process_captions
from mllm.dataset.utils import save_to_json, save_to_jsonl, read_json_file, read_jsonl_file, read_txt_file 
from mllm.utils.standalone_eval.eval import compute_mr_ap

base_file = '/share_ssd/tianjirui/dtos_output/stage1/exp97/checkpoint-50000/eval_top5'  # /eval_top1  /checkpoint-70000

data_file = os.path.join(base_file, 'multitest_qv_test_prediction.pth')
meta_file = '/share_ssd/tianjirui/QVHighlishts/highlight_test_release.jsonl'
save_file = os.path.join(base_file, 'hl_test_submission.jsonl')


val_data_file = os.path.join(base_file, 'multitest_qv_val_prediction.pth')
val_meta_file = '/share_ssd/tianjirui/QVHighlishts/highlight_val_release.jsonl'
val_save_file = os.path.join(base_file, 'hl_val_submission.jsonl')
eval_topk = 5
val_map_save_file = os.path.join(base_file, f'qv_val_mAP_top{eval_topk}.json')


def get_formatting_data(data, meta_data, topk = None):
    submissions = []
    # build qid map dict
    pred_cap_map = {d['captions'][0] : d for d in data}
    
    for i, md in enumerate(meta_data): # 遍历测试数据更严格
        # find rec
        caption = [md['query']]
        item = pred_cap_map[pre_process_captions(caption)[0]]
        avail_num = item['pred_num']
        rec = item['pred_ts'].to(dtype=torch.float32).squeeze(0).numpy()[:avail_num]
        if topk is not None:
            select_num = min(topk, avail_num)
            rec = rec[:select_num]
        rec[:, :2] = rec[:, :2].astype(np.int_)
        rec[:, 2] = rec[:, 2].astype(np.float_)
        rec = rec.tolist()
        
        out = {}
        out["qid"] = int(md['qid'])
        out["query"] = md["query"]
        out["vid"] = md["vid"]
        out["pred_relevant_windows"] = rec
        out["pred_saliency_scores"] = [1.0] * len(rec) # dummy
        submissions.append(out)
    return submissions
    

# save formatting data (test)
data = torch.load(data_file, map_location=torch.device('cpu'))
meta_data = read_jsonl_file(meta_file)
save_data = get_formatting_data(data, meta_data)
save_to_jsonl(save_data, save_file)

# save formatting data (val)
val_data = torch.load(val_data_file, map_location=torch.device('cpu'))
val_meta_data = read_jsonl_file(val_meta_file)
val_save_data = get_formatting_data(val_data, val_meta_data)
save_to_jsonl(val_save_data, val_save_file)

# evaluate qv_val result
val_data = torch.load(val_data_file, map_location=torch.device('cpu'))
_ground_truth = read_jsonl_file(val_meta_file)
_submission = get_formatting_data(val_data, _ground_truth, topk = eval_topk)
iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
save_to_json(iou_thd2average_precision, val_map_save_file)

print('qv_val mAP@.5: ', iou_thd2average_precision['0.5'])
print('qv_val mAP@.75: ', iou_thd2average_precision['0.75'])