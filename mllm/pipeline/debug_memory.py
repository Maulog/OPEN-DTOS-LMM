import os
import sys
import logging
import pathlib
import typing
import warnings

import torch
import torch.cuda
from tqdm import tqdm

from mllm.config import prepare_args
from mllm.models.builder import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ], # 将日志消息发送至控制台
)


def main():
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    print_trainable_params(model) # 打印出可训练的参数数量

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs # {padding, max_length}
    trainer_cls, data_collator_dict, trainer_callbacks = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)

    def collate_with_data_collector(batch):
        data_collector = data_collator_dict['train_collator']
        processed_batch = data_collector(batch)
        # 如果需要，这里还可以进一步整理批次数据，例如堆叠张量等
        return processed_batch  # 使用默认的collate逻辑来堆叠样本到一个批次，来自torch.utils.data._utils.collate.default_collate


    # 创建DataLoader实例
    data_loader = DataLoader(dataset['train'], 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4, # 4
                            collate_fn=collate_with_data_collector)
    
    for epoch in range(1):
        for i, item in tqdm(enumerate(data_loader)):
            item = item
            print(i)
            pass
            # print(f"{i}")

# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
