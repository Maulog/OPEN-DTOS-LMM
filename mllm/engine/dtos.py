import os
from typing import Optional
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.distributed as dist


from transformers.trainer import unwrap_model
from transformers import TrainerCallback, TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy, EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.utils import logging

from .base_engine import TrainerForMMLLM

logger = logging.get_logger(__name__)


class DtosLocTrainer(TrainerForMMLLM):
    def _save(self, output_dir: Optional[str] = None, state_dict=None): # 最后一次保存会进来
        # Save the model
        _state_dict = state_dict
        model_to_save = unwrap_model(self.model)
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            _state_dict = model_to_save.state_dict()

        # output_dir一开始命名为tmp开头的之后会将其修改为checkpoint开头
        model_to_save.rec_token_projector.merge_to_embedding()
        weight_to_save = {}
        lora_to_save = {}
        keys_to_match = ['rec_decoder',
                         'rec_encoder',
                         'rec_score_head',
                         'lm_head',
                         'rec_token_projector.orig_emb', 
                         'lora']
        for k, v in state_dict.items(): # 看一下state_dict
            if any(key_match in k for key_match in keys_to_match):
                if 'lora' in k:
                    lora_to_save[k] = v
                else:
                    weight_to_save[k] = v
        
        # 将其保存为一个文件不进行区分（不然太散了
        dtos_loc_dir = os.path.join(output_dir, "dtos_loc")
        os.makedirs(dtos_loc_dir, exist_ok=True)
        torch.save(weight_to_save, os.path.join(dtos_loc_dir, f'dtos_loc.bin'))
        
        # lora_dir = os.path.join(output_dir, "lora") # 可能不需要单独保存，会在后面trainer进行保存
        # os.makedirs(lora_dir, exist_ok=True)
        # torch.save(lora_to_save, os.path.join(lora_dir, f'lora.bin'))
        
        model_to_save.config.save_pretrained(output_dir)
        
        # 是因为添加了lora会识别为认识的类，然后不走模型的save_pretrained
        # 此处不加lora会报错
        super(DtosLocTrainer, self)._save(output_dir, _state_dict) # 重写save_pretrained避开llavameta


class DtosPrintLossCallback(TrainerCallback): # 将损失值更新进log_history
    def on_log(self, args: TrainingArguments, state, control, logs, **kwargs): # type: ignore
        latest_log = state.log_history[-1]
        if state.other_information is not None:
            logs.update(state.other_information)
            latest_log.update(state.other_information)
        # if logs and "other_information" in logs:
        #     other_information = logs["other_information"]
            



class DtosSegTrainer(TrainerForMMLLM):
    def _save(self, output_dir: Optional[str] = None, state_dict=None): # 最后一次保存会进来
        # Save the model
        _state_dict = state_dict
        model_to_save = unwrap_model(self.model)
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            _state_dict = model_to_save.state_dict()

        # output_dir一开始命名为tmp开头的之后会将其修改为checkpoint开头
        model_to_save.seg_token_projector.merge_to_embedding()
        weight_to_save = {}
        lora_to_save = {}
        keys_to_match = [#'seg_decoder',
                         'box_decoder',
                         'seg_score_head',
                         'lm_head',
                         'seg_token_projector.orig_emb', 
                         'lora']
        for k, v in state_dict.items(): # 看一下state_dict
            if any(key_match in k for key_match in keys_to_match):
                if 'lora' in k:
                    lora_to_save[k] = v
                else:
                    weight_to_save[k] = v
        
        # 将其保存为一个文件不进行区分（不然太散了
        dtos_loc_dir = os.path.join(output_dir, "dtos_seg")
        os.makedirs(dtos_loc_dir, exist_ok=True)
        torch.save(weight_to_save, os.path.join(dtos_loc_dir, f'dtos_seg.bin'))
        
        model_to_save.config.save_pretrained(output_dir)
        
        # 是因为添加了lora会识别为认识的类，然后不走模型的save_pretrained
        # 此处不加lora会报错
        super(DtosSegTrainer, self)._save(output_dir, _state_dict) # 重写save_pretrained避开llavameta
        
    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

            Works both with or without labels.
            """
            args = self.args

            prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

            # if eval is called w/o train, handle model prep here
            if self.is_deepspeed_enabled and self.deepspeed is None:
                _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

            model = self._wrap_model(self.model, training=False, dataloader=dataloader)

            if len(self.accelerator._models) == 0 and model is self.model:
                model = (
                    self.accelerator.prepare(model)
                    if self.is_deepspeed_enabled
                    else self.accelerator.prepare_model(model, evaluation_mode=True)
                )

                if self.is_fsdp_enabled:
                    self.model = model

                # for the rest of this function `model` is the outside model, whether it was wrapped or not
                if model is not self.model:
                    self.model_wrapped = model

                # backward compatibility
                if self.is_deepspeed_enabled:
                    self.deepspeed = self.model_wrapped

            # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
            # while ``train`` is running, cast it to the right dtype first and then put on device
            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)

            batch_size = self.args.eval_batch_size

            logger.info(f"***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.
            eval_dataset = getattr(dataloader, "dataset", None)

            if args.past_index >= 0:
                self._past = None

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            losses_host = None
            preds_host = None
            labels_host = None
            inputs_host = None

            # losses/preds/labels on CPU (final containers)
            all_losses = None
            all_preds = []
            all_labels = []
            all_inputs = None
            all_extra_list = []
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                main_input_name = getattr(self.model, "main_input_name", "input_ids")
                inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None
                # 这里的inputs_decode也必须是tensor类型，无论外面是什么包装
                
                # 首先放到相应的设备上，再同步不同进程内的信息，然后收集每一轮的预测
                # 这里放弃直接gather整个mask，因为同样会遇到不同hw的问题（如果继续也需要解决），同时直接保存
                # gather_logits_list = [None for _ in range(dist.get_world_size())]
                # for i, _ in enumerate(gather_logits_list):
                #     gather_logits_list[i] = torch.zeros_like(logits, device='cpu')
                # dist.all_gather(gather_logits_list, logits) # 这里先搁置一下，因为也可能遇到hw不一样的情况
                # gather_labels_list = [None for _ in range(dist.get_world_size())]
                # dist.all_gather(gather_labels_list, labels)
                
                # logits = self.gather_function((logits)) # 这里感觉如果遇到两个视频hw不一样仍然不可以？？
                # labels = self.gather_function((labels)) # 超显存！！还没放到cpu
                
                # all_preds.append(logits)
                # all_labels.append(labels)
                
                logits = logits.cpu().numpy()
                labels = labels.cpu().numpy()
                
                if self.compute_metrics is not None: # 进行单个结果的评估
                    if args.include_inputs_for_metrics: # 这里尝试直接用一个函数取出结果，不全部送入
                        self.compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels, inputs=None)
                        )
                    else:
                        self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
                
                
            
            # After all calls to `.gather_function`, reset to `gather_for_metrics`:
            self.gather_function = self.accelerator.gather_for_metrics
            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")
                
            # Number of samples   # 从这里开始操作完数据
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
                num_samples = eval_dataset.num_examples
            else:
                if has_length(dataloader):
                    num_samples = self.num_examples(dataloader)
                else:  # both len(dataloader.dataset) and len(dataloader) fail
                    num_samples = observed_num_examples
            if num_samples == 0 and observed_num_examples > 0:
                num_samples = observed_num_examples

            # Metrics!
            # 这里首先汇总结果，然后送入metrics进行评估
            all_ious, all_boundary_f = self.compute_metrics.get_all_metrics()
            
            global_all_iou_lists = [None for _ in range(dist.get_world_size())]
            global_all_boundary_f_lists = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(global_all_iou_lists, all_ious) # 将不同卡上的信息收集
            dist.all_gather_object(global_all_boundary_f_lists, all_boundary_f)
            
            global_all_iou_lists = sum(global_all_iou_lists, []) # 这里处理返回的列表
            global_all_boundary_f_lists = sum(global_all_boundary_f_lists, [])
            
            metrics = self.compute_metrics.return_metrics(global_all_iou_lists, global_all_boundary_f_lists)
            
            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics) # 转换为python的数据类型

            if all_losses is not None:
                metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            if hasattr(self, "jit_compilation_time"):
                metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

            
