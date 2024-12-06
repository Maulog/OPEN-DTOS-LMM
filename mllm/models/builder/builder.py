# 一个整合加载预训练模型的函数，包括加载各阶段的模型参数
from typing import Dict, Any, Tuple

from torch import nn

from mllm.engine import LOAD_PRETRAINED, BOXES_PROCESSOR



PREPROCESSOR = Dict[str, Any]

def prepare_target_processor(
        model,  # multimodal llm
        preprocessor: Dict[str, Any],
        model_args,
        training_args,
):
    if not hasattr(model_args, 'target_processor'): # 之后这里可能需要删除
        return model, preprocessor

    target_processor = {}
    if 'boxes' in model_args['target_processor']:
        boxes_cfg = model_args['target_processor']['boxes']
        boxes_processor = BOXES_PROCESSOR.build(boxes_cfg)
        target_processor['boxes'] = boxes_processor
        if hasattr(boxes_processor, "post_process_model_tokenizer"): # 暂时没用到
            model, preprocessor = boxes_processor.post_process_model_tokenizer(
                model, preprocessor, model_args, training_args,
            )
    preprocessor['target'] = target_processor
    return model, preprocessor

def load_pretrained(model_args_, training_args_) -> Tuple[nn.Module, PREPROCESSOR]:
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    type_ = 'load_pretrained_' + model_args_.type
    model, preprocessor = LOAD_PRETRAINED.build(dict(type=type_, model_args=model_args_, training_args=training_args_)) # 创建相应加载的函数
    model, preprocessor = prepare_target_processor(model, preprocessor, model_args_, training_args_) # preprocessor是一个包含图像、文本、对话样式、候选框的处理器
    return model, preprocessor
