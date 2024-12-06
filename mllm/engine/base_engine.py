import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINER_STATE_NAME

from llava.mm_utils import process_images
from mllm.dataset.utils.io import save_to_jsonl
from mllm.dataset.utils.rvos_saver import MaskSaver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class TrainerDifferentCollatorMixin: # mixin为混入类，临时使用不同数据集加载器
    def __init__(self,
                 *args,
                 train_collator: Optional[DataCollator] = None, # type: ignore
                 eval_collator: Optional[DataCollator] = None, # type: ignore
                 test_collator: Optional[DataCollator] = None, # type: ignore
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
            warnings.warn('[WARNING!!!] use different collator for eval and test. but maybe do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.) u should'
                          'check your code and know exactly what u are doing.')
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator # 没定义用训练的
        self._test_collator = test_collator if test_collator is not None else self._eval_collator # 没定义用验证的
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
        self.tmp_list = [] # 用于保存额外信息,当前用于保存captions
        self.tmp_dict = {}
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_train_dataloader(self) -> DataLoader: # 临时替换data_collator作为返回
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader


# noinspection DuplicatedCode
class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs) # 递归处理每一项数据，处理数据类型和设备，并兼容deepspeed
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None: # 模型具有过去状态，可能是transformer具有的记忆机制
            inputs["mems"] = self._past # 将过去的状态传入模型
        return inputs

    def training_step(self, model, inputs): # modify training step from trainer
        # 注入了自定义行为保存额外信息
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            other_info = outputs[-1]
        
            if other_info is not None:
                self.state.other_information = other_info

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        
        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step( # 实际进行一步推理（没有梯度
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom behavior.

        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only: # 不使用generate预测或者只计算loss
            return super().prediction_step( # 设置了predict_with_generate不会进
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy() # 生成函数的参数
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = ( # 推理一定使用zero3所以会进这里
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs: # 把标签中的labels剔除作为generate的生成
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys()) # 打印使用的generate参数有哪些
        
        with torch.inference_mode(): # 注意看下这里，可能使用一个分支语句来区分两个阶段
            with self.compute_loss_context_manager(): 
                output_seq, pred_locs = self.model.generate(**gen_kwargs) # 此处应该也强制dtos_seg也输出一样的形式
                

        loss = None
        

        if has_labels: # postprocess, nms
            if "timestamps" in inputs: # mr task # 用字典的方式不能分开，尝试把所有tensor变为float
                # TODO: 将预测的结果保存下来
                if torch.all(pred_locs==-100):
                    pred_num = torch.tensor(0).to(device=pred_locs[0].device)
                else:
                    pred_num = torch.tensor(pred_locs.shape[1]).to(device=pred_locs[0].device)
                
                generated_tokens = { # 此处只能是tensor类型,不能是str类型
                    'pred_ts': pred_locs,
                    'pred_num': pred_num,
                }
                labels = {
                    'label_ts': torch.stack(inputs["timestamps"]), 
                    'label_num':torch.tensor(len(inputs["timestamps"][0])).to(device=inputs["timestamps"][0].device),
                }
                save_dict = {
                    'pred_ts': pred_locs,
                    'pred_num': pred_num,
                    'captions': inputs['captions'], # 辅助后期查找
                    'source': inputs['data_dict']['source'],
                    'video': inputs['data_dict']['video'],
                    'label_ts': torch.stack(inputs["timestamps"]), 
                    'label_num': torch.tensor(len(inputs["timestamps"][0])).to(device=inputs["timestamps"][0].device)
                }
                self.tmp_list.append(save_dict)
                # self.tmp_dict.update(save_dict)
                
            elif "masks" in inputs: # rec bounding box eval ，计划在stage2使用，待完善
                gt_masks = inputs['data_dict']['all_masks'] # 这里可以不传递，传递相关的路径应该就行
                pred_masks = pred_locs['pred_masks']
                
                if pred_masks is not None: # 处理mask格式
                    frms_mask_list = []
                    vid_len = len(pred_masks)
                    for frm_idx in range(vid_len):
                        all_obj_mask_per_frm = pred_masks[frm_idx]
                        cur_frm_mask = torch.zeros(gt_masks[0].shape, device=gt_masks.device)
                        for cur_obj_mask in all_obj_mask_per_frm.values():
                            cur_frm_mask += cur_obj_mask
                        frms_mask_list.append(cur_frm_mask)
                    pred_masks = torch.stack(frms_mask_list)
                else:
                    pred_masks = torch.zeros(gt_masks.shape, device=gt_masks.device)
        
                # format variables to continue
                # generated_tokens = {
                #     # 'pred_prompts': pred_prompts,
                #     'pred_masks': pred_masks,
                # }
                # labels = { # 这里尝试返回对应字符串和掩码，使其在计算metrics的时候找到对应路径
                #     # 'vid_path': inputs['video_dir'],
                #     'gt_masks': inputs['data_dict']['all_masks'], # 这里不行因为不同hw不好补齐
                # }
                
                generated_tokens = pred_masks
                labels = inputs['data_dict']['all_masks']
                
                pred_prompts = pred_locs['tgt_frm_res'] # 这里改设备到cpu
                
                for i, tgt_num in enumerate(pred_prompts.keys()):
                    prompt = pred_prompts[tgt_num]
                    for key, value in prompt.items():
                        pred_prompts[tgt_num][key] = value.to(device='cpu') if type(value) == torch.Tensor else value
                
                save_dict = { 
                    'lmm_pred': pred_prompts, 
                    'choosed_tgt_id': pred_locs['choosed_tgt_frm'],
                    'filter_boxes': pred_locs['filter_boxes'],
                    'filter_scores': pred_locs['filter_scores'],
                    'captions': inputs['captions'], # 辅助后期查找，不用exp_id是防止用其他的数据集没有
                    'exp_id': inputs['data_dict']['exp_id'] if inputs['data_dict']['exp_id'] is not None else None,
                    'source': inputs['data_dict']['source'],
                    'video': inputs['data_dict']['video'],
                    'video_dir': inputs['video_dir'],
                    'orig_size': inputs['orig_size'].to(device='cpu').numpy(),
                    'frames_idx': inputs['frames_idx'].to(device='cpu').numpy(), # 输入模型的帧编号
                    # 这里考虑是否存预测的mask和标签的mask，不保存，太大
                    # 'gt_masks': gt_masks.to(device='cpu').numpy(),
                    # 'pred_masks': pred_masks.to(device='cpu').numpy(),
                }
                self.tmp_list.append(save_dict)
                
                # 此处保存png图片（仅在test数据集中
                if 'test' == inputs['data_dict']['dataset_type'] or 'val' == inputs['data_dict']['dataset_type']:
                    save_path = self.args.output_dir
                    dataset_name = inputs['data_dict']['source']
                    dataset_type = inputs['data_dict']['dataset_type']
                    video_name = inputs['data_dict']['video']
                    exp_id = inputs['data_dict']['exp_id']
                    frame_names = inputs['data_dict']['frame_names']
                    
                    # save_folder = os.path.join(save_path, video_name, exp_id)
                    # os.makedirs(save_folder, exist_ok=True)
                    
                    # 处理mask
                    rvos_saver = MaskSaver(save_path, dataset_name, dataset_type, video_name, exp_id)
                    rvos_saver.save_all_masks(pred_masks, frame_names)
                    
            else: # video2text gen task eval 计划使用双向的时候用
                labels = inputs["labels"]
        else: # 没有标签
            labels = None
        
        return loss, generated_tokens, labels

    def de_transform_mask(self, orgw, orgh, mask): # 原始高宽，输入mask，还原处理后的mask
        long_side = max(orgw, orgh)
        short_side = min(orgw, orgh)
        pad = (long_side - short_side) // 2 # 长边与短边的差值/2
        mask = F.interpolate(mask, [long_side, long_side], mode="bilinear", align_corners=False)
        mask = mask > 0
        mask[mask>0] = 1
        mask[mask<=0] = 0
        if orgw < orgh:
            mask = mask[..., :, pad: short_side + pad] # 高度大于宽度，截取宽度
        else:
            mask = mask[..., pad: short_side + pad, :] # 宽度大于高度，截取高度
        # mask = mask.transpose(2, 3)
        # print(mask.shape)
        return mask.squeeze(1)

    def tensor2token(self, tensor_list): # 经过tokenizer转换为token，返回ids
        lst = [str(tensor.cpu().tolist()) for tensor in tensor_list]
        tokens = self.tokenizer(lst, return_tensors="pt", add_special_tokens=False, padding="longest")
        return tokens.input_ids.to(tensor_list[0].device)

    def _logging_generate_kwargs(self, keys): # 处理传入的_generate_kwargs
        if not hasattr(self, '_generate_kwargs'):
            self._generate_kwargs = None
        if self._generate_kwargs != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def save_prediction(self, predict_results, file_key_prefix='predict'): # 待修改
        # torch.distributed.is_initialized() # 这里全是true
        all_tmp_lists = [None for _ in range(torch.distributed.get_world_size())]
        all_tmp_dicts = [None for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather_object(all_tmp_lists, self.tmp_list) # 将卡上的信息收集到all_tmp_lists中
        torch.distributed.all_gather_object(all_tmp_dicts, self.tmp_dict)

        # Check if this is the main process
        if self.is_world_process_zero():
            # Flatten and combine the lists and dictionaries
            combined_tmp_list = sum(all_tmp_lists, [])
            combined_tmp_dict = {}
            for d in all_tmp_dicts:
                combined_tmp_dict.update(d)
        
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # 用这个方法读取np.load(a, allow_pickle=True).item()得到字典
            if self.tmp_list != []:
                # save_to_jsonl(self.tmp_list, os.path.join(self.args.output_dir, f"{file_key_prefix}_prediction.jsonl")) # 不能保存tensor
                torch.save(combined_tmp_list, os.path.join(self.args.output_dir, f"{file_key_prefix}_prediction.pth"))
            if self.tmp_dict != {}:
                torch.save(combined_tmp_dict, os.path.join(self.args.output_dir, f"{file_key_prefix}_prediction.pth"))
            
            # 这里区分两个阶段的保存！！
            if 'stage1' in self.args.output_dir:
                preds, targets = predict_results.predictions, predict_results.label_ids
                preds, targets = deepcopy(preds), deepcopy(targets)
                logger.warning(f"results saved to {self.args.output_dir}")
                logger.warning(f"preds shape: {preds['pred_ts'].shape}. targets shape: {targets['label_ts'].shape}")
            elif 'stage2' in self.args.output_dir:
                logger.warning(f"results saved to {self.args.output_dir}")
            
            # decode text and save to json takes forever for big test set
            os.makedirs(self.args.output_dir, exist_ok=True)
            
        
        torch.distributed.barrier() # 同步所有卡，之后进行清空
        self.tmp_list = [] # 清空临时信息
        self.tmp_dict = {} # 清空临时信息
        
    # refer: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call) # transformer库的保存方式

    def plot_loss(self) -> None: # 生成trainer的损失函数图
        if not self.is_world_process_zero():
            return

        training_args = self.args
        FIGURE_NAME = "trainer_state.png"
        import matplotlib.pyplot as plt
        data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
        if 'stage2' not in training_args.output_dir:
            train_steps, train_losses, train_iou_losses = [], [], []
            train_l1_losses, train_error_losses, train_langmodel_losses = [], [], []
            train_label_losses, train_cycle_losses = [], []
            for i in range(len(data["log_history"]) - 1):
                train_steps.append(data["log_history"][i]["step"])
                train_losses.append(data["log_history"][i]["loss"])
                train_iou_losses.append(data["log_history"][i]["iou_loss"])
                train_l1_losses.append(data["log_history"][i]["l1_loss"])
                train_error_losses.append(data["log_history"][i]["error_loss"])
                train_langmodel_losses.append(data["log_history"][i]["lang_model_loss"])
                train_label_losses.append(data["log_history"][i]["label_loss"])
                train_cycle_losses.append(data["log_history"][i]["cycle_loss"])
            plt.figure()
            # plt.plot(train_steps, train_losses, label="total loss")
            # plt.plot(train_steps, train_langmodel_losses, label="langmodel loss")
            plt.plot(train_steps, train_l1_losses, label="l1 loss")
            plt.plot(train_steps, train_iou_losses, label="iou loss")
            plt.plot(train_steps, train_label_losses, label="label loss")
            plt.plot(train_steps, train_cycle_losses, label="cycle loss")
            # plt.plot(train_steps, train_error_losses, label="error loss")
        else:
            train_steps, train_losses, train_box_giou_losses = [], [], []
            train_box_l1_losses, train_error_losses, train_langmodel_losses = [], [], []
            train_label_losses, train_sam_losses = [], []
            for i in range(len(data["log_history"]) - 1):
                train_steps.append(data["log_history"][i]["step"])
                train_losses.append(data["log_history"][i]["loss"])
                train_box_giou_losses.append(data["log_history"][i]["box_giou_loss"])
                train_box_l1_losses.append(data["log_history"][i]["box_l1_loss"])
                train_error_losses.append(data["log_history"][i]["error_loss"])
                train_langmodel_losses.append(data["log_history"][i]["lang_model_loss"])
                train_label_losses.append(data["log_history"][i]["label_loss"])
                train_sam_losses.append(data["log_history"][i]["sam_loss"])
            plt.figure()
            # plt.plot(train_steps, train_losses, label="total loss")
            # plt.plot(train_steps, train_langmodel_losses, label="langmodel loss")
            plt.plot(train_steps, train_box_l1_losses, label="box l1 loss")
            plt.plot(train_steps, train_box_giou_losses, label="box giou loss")
            plt.plot(train_steps, train_label_losses, label="label loss")
            # plt.plot(train_steps, train_sam_losses, label="sam loss")
            plt.plot(train_steps, train_error_losses, label="error loss")
        
        plt.legend()
        plt.title("training loss of {}".format(training_args.output_dir))
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
        print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


class Seq2SeqDataCollator(DataCollatorForSeq2Seq): 
    # 继承huggingface的collator，重写一个padding的补全方法，推理训练模式
    def __init__(
            self,
            inference_mode: bool = False,
            **kwargs,
    ):
        self.inference_mode = inference_mode
        self.text_keys = ['input_ids', 'labels', 'attention_mask']
        super().__init__(**kwargs)

    def __call__(self, feature: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
        # evaluation/inference adopts left-padding while training adopts right-padding
        text_features = [{k: feature[k] for k in self.text_keys if k in feature}]
        # 推理模式向左补齐，训练模式向右补齐
        if self.inference_mode:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side
        else:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'right'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side

        return text_features


class Seq2Seq2LocCollatorWithImage(Seq2SeqDataCollator): # 用于处理带有图片输入的
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)
        # sometimes there is either no location input or output in the current batch
        # which will make some parameters untrained in the batch.
        # use a mock annotation to prevent error
        # self.mock = torch.load("mock.pth") # 模拟数据填充批次，防止训练时因缺少数据出现错误

    # noinspection PyMethodMayBeStatic
    def _image_process(self, images: List) -> torch.Tensor: # 处理图片，暂时没用到
        # images = torch.stack(images, dim=0)
        return images
    
    def _video_process(self, video: List) -> torch.Tensor: # 处理视频
        # video = process_images(video, image_processor=, model_cfg=)
        # video = torch.stack(video, dim=0)
        return video
    

    def __call__(self, data_dict: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]: # 此处处理sam、mask
        # if not self.inference_mode and not ("masks_sam" in data_dict[0]): # 添加mock的条件
        #     # 不在推理模式下，且特征字典中没有masks_sam，防止在此处处理出错？？
        #     data_dict.append(self.mock)
        assert len(data_dict) == 1, "only support batch size 1"
        data_dict = data_dict[0]
        
        ret = super().__call__(data_dict['conversation'], return_tensors)
        # data_dict.update(super().__call__(data_dict['conversation'], return_tensors))
        if data_dict['im_src'] is not None: # 使用原视频
            data_dict['im_src'] = self._image_process(data_dict['im_src']) # 空操作
            data_dict['vi_src'] = self._video_process(data_dict['vi_src'])
            
            
        # # 不删除，直接解析为forward的入口参数
        ret['im_src'] = data_dict.pop('im_src')
        ret['vi_src'] = data_dict.pop('vi_src')
        ret['vi_feat'] = data_dict.pop('vi_feat')
        ret['im_feat'] = data_dict.pop('im_feat')
        
        ret['timestamps'] = data_dict.pop('timestamps')
        ret['captions'] = data_dict.pop('captions')
        ret['norm_timestamps'] = data_dict.pop('norm_timestamps')
        ret['reverse_list'] = data_dict.pop('reverse_list')
        
        ret['clip_length'] = data_dict.pop('clip_length') # data_dict['clip'][1] - data_dict['clip'][0] 
        ret['data_dict'] = data_dict # 将剩余的数据存到data_dict中
        
        ret['output_hidden_states'] = True
        ret['output_attentions'] = True
        
        
        return ret
        
            
class Seq2Seq2SegCollatorWithImage(Seq2SeqDataCollator): # 用于处理带有图片输入的
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)

    # noinspection PyMethodMayBeStatic
    def _image_process(self, images: List) -> torch.Tensor: # 处理图片，暂时没用到
        return images
    
    def _video_process(self, video: List) -> torch.Tensor: # 处理视频
        return video
    

    def __call__(self, data_dict: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]: # 此处处理sam、mask
        assert len(data_dict) == 1, "only support batch size 1"
        data_dict = data_dict[0]
        
        ret = super().__call__(data_dict['conversation'], return_tensors)
        # data_dict.update(super().__call__(data_dict['conversation'], return_tensors))
        if data_dict['im_src'] is not None: # 使用原视频
            data_dict['im_src'] = self._video_process(data_dict['im_src']) # 空操作
            # data_dict['im_src'] = self._image_process(data_dict['im_src'])
            
            
        # # 不删除，直接解析为forward的入口参数
        ret['im_src'] = data_dict.pop('im_src')
        # ret['vi_src'] = data_dict.pop('vi_src')
        ret['vi_feat'] = data_dict.pop('vi_feat')
        ret['im_feat'] = data_dict.pop('im_feat')
        ret['orig_image_list'] = data_dict.pop('orig_image_list')
        
        ret['orig_size'] = data_dict.pop('orig_size')
        ret['frames_idx'] = data_dict.pop('frames_idx')
        ret['video_dir'] = data_dict.pop('video_dir')
        ret['captions'] = data_dict.pop('captions')
        
        ret['tgt_frm_idx'] = data_dict.pop('tgt_frm_idx')
        ret['norm_bbox'] = data_dict.pop('norm_bbox')
        ret['masks'] = data_dict.pop('masks')
        ret['valid'] = data_dict.pop('valid')
        # ret['objs_masks_dict'] = data_dict.pop('objs_masks_dict')
        ret['tgt_norm_bbox_list'] = data_dict.pop('tgt_norm_bbox_list')
        
        
        ret['data_dict'] = data_dict # 将剩余的数据存到data_dict中
        
        ret['output_hidden_states'] = True
        ret['output_attentions'] = True
        
        
        return ret