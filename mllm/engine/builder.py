from functools import partial
from typing import Tuple, Dict, Any, Type

from transformers.trainer import DataCollator

from .dtos import DtosSegTrainer, DtosLocTrainer, DtosPrintLossCallback
from .base_engine import TrainerForMMLLM, Seq2Seq2LocCollatorWithImage, Seq2Seq2SegCollatorWithImage

TYPE2TRAINER = { # TODO：use registry
    'dtos_loc': DtosLocTrainer,  # 或者只用一个DtosTrainer
    'dtos_seg': DtosSegTrainer, 
    'dtos_debug': DtosLocTrainer,
}
TYPE2CALLBACK = {
    'dtos_loc': DtosPrintLossCallback,
    'dtos_seg': DtosPrintLossCallback,
    'dtos_debug': DtosPrintLossCallback,
}
TYPE2COLLATOR = {
    'dtos_loc': Seq2Seq2LocCollatorWithImage,
    'dtos_seg': Seq2Seq2SegCollatorWithImage,
    'dtos_debug': Seq2Seq2LocCollatorWithImage,
}


def prepare_trainer_collator(
        model_args,
        preprocessor: Dict[str, Any],
        collator_kwargs: Dict[str, Any]
) -> Tuple[Type[TrainerForMMLLM], Dict[str, DataCollator]]: # type: ignore
    type_ = model_args.type
    trainer_cls = TYPE2TRAINER.get(type_) # 根据name返回相应的trainer
    trainer_callbacks = [TYPE2CALLBACK.get(type_)]
    collator = TYPE2COLLATOR.get(type_)
    data_collator_func = partial(  # 创建一个新的函数or对象，将一个函数的某些参数提前绑定，只用传递剩下的参数即可
        collator,
        preprocessor=preprocessor,
        **collator_kwargs,
    )
    data_collator_dict = {
        "train_collator": data_collator_func(inference_mode=False),
        "eval_collator": data_collator_func(inference_mode=True),
    }
    return trainer_cls, data_collator_dict, trainer_callbacks
