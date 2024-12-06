import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
from argparse import SUPPRESS

import datasets
import transformers
from mmengine.config import Config, DictAction
from transformers import HfArgumentParser, set_seed, add_start_docstrings
from transformers import Seq2SeqTrainingArguments as HFSeq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@dataclass
@add_start_docstrings(HFSeq2SeqTrainingArguments.__doc__) # 添加文档说明
class Seq2SeqTrainingArguments(HFSeq2SeqTrainingArguments):
    do_multi_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the multi-test set."})


def prepare_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument( # 自定义参数解析，为了不修改配置文件只修改命令行命令设计
        '--cfg-options',
        nargs='+', # 一个或多个参数
        action=DictAction, # 接受到的参数存储为字典
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    hf_parser, required = block_required_error(hf_parser) # 阻止必须参数的检查

    args, unknown_args = parser.parse_known_args(args) # 将命令行参数解析为Namespace对象
    known_hf_args, unknown_args = hf_parser.parse_known_args(unknown_args) # 解析hg的参数
    if unknown_args:
        raise ValueError(f"Some specified arguments are not used "
                         f"by the ArgumentParser or HfArgumentParser\n: {unknown_args}")

    # load 'cfg' and 'training_args' from file and cli
    cfg = Config.fromfile(args.config) # mmengine从文件中加载配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options) # 将命令行的参数合并到cfg当中
    training_args = cfg.training_args
    training_args.update(vars(known_hf_args)) # var获取对象的属性和属性值，将known_hf_args的属性和属性值更新到training_args中
    
    # data_args prepare, broadcast args to all data_args
    for type_name in ['train', 'validation', 'test', 'multitest']:
        if cfg.data_args[type_name] is not None:
            if type_name == 'multitest':
                for key in cfg.data_args[type_name]: # 这里以后冗余了可以用循环
                    cfg.data_args[type_name][key].cfg.use_video = cfg.data_args.use_video
                    cfg.data_args[type_name][key].cfg.n_frms = cfg.data_args.n_frms
                    if 'sampling_type' in cfg.data_args[type_name][key].cfg:
                        cfg.data_args[type_name][key].cfg.sampling_type = cfg.data_args.sampling_type
                        cfg.data_args[type_name][key].cfg.dynamic_path = cfg.data_args.dynamic_path
            else:
                for i, _ in enumerate(cfg.data_args[type_name].cfgs):
                    cfg.data_args[type_name].cfgs[i].use_video = cfg.data_args.use_video
                    cfg.data_args[type_name].cfgs[i].n_frms = cfg.data_args.n_frms
                    if 'sampling_type' in cfg.data_args[type_name].cfgs[i]:
                        cfg.data_args[type_name].cfgs[i].sampling_type = cfg.data_args.sampling_type
                        cfg.data_args[type_name].cfgs[i].dynamic_path = cfg.data_args.dynamic_path

    # check training_args require
    req_but_not_assign = [item for item in required if item not in training_args]
    if req_but_not_assign:
        raise ValueError(f"Requires {req_but_not_assign} but not assign.")

    # update cfg.training_args
    cfg.training_args = training_args

    # initialize and return
    training_args = Seq2SeqTrainingArguments(**training_args)
    training_args = check_output_dir(training_args)

    # logging
    if is_main_process(training_args.local_rank):
        to_logging_cfg = Config() # mmengine
        to_logging_cfg.model_args = cfg.model_args
        to_logging_cfg.data_args = cfg.data_args
        to_logging_cfg.training_args = cfg.training_args
        logger.info(to_logging_cfg.pretty_text)

    # setup logger   将整个系统中logger、transformers、datasets的日志级别设置为training_args.get_process_log_level()
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()  # 获取当前进程的日志级别
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format() # 显式格式化输出
    # setup_print_for_distributed(is_main_process(training_args))

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, fp16 training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return cfg, training_args


def block_required_error(hf_parser: HfArgumentParser) -> Tuple[HfArgumentParser, List]:
    required = []
    # noinspection PyProtectedMember
    for action in hf_parser._actions:
        if action.required:
            required.append(action.dest)
        action.required = False
        action.default = SUPPRESS
    return hf_parser, required


def check_output_dir(training_args):
    # Detecting last checkpoint.
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        # 在训练模式下，且output_dir存在，且不覆盖output_dir
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # 如果output_dir存在，但是没有checkpoint，则报错
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None: 
            # 如果存在checkpoint，但是没有指定resume_from_checkpoint
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return training_args


if __name__ == "__main__":
    _ = prepare_args()
