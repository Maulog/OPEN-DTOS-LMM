# DTOS: Dynamic Time Object Sensing with LMM

<img src="assert\motivation.png#pic_center" alt="motivation" style="zoom:38%;" />

## Overview

![model](assert\model.png)

DTOS demonstrates exceptional capability in flexibly localizing multiple spatiotemporal targets based on user provided textual instructions.

## Setup

### Install Dependencies

```shell
# CUDA117
conda create -n dtos python==3.10
conda activate dtos

pip install -r requirement.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') # refer VILA
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
```

### Download Pretrained Models

Please download 

[VILA]: https://huggingface.co/collections/Efficient-Large-Model/vila-on-pre-training-for-visual-language-models-65d8022a3a52cd9bcd62698e

,

[SAM2]: https://huggingface.co/models?search=facebook/sam2

 weight checkpoints. We haven't use SAM2.1 checkpoint work yet.

### Dataset Preparation

Please download these original datasets.

**Moment Retrieval**

- [DiDeMo]: https://github.com/lisaanne/localizingmoments?tab=readme-ov-file#localizing-moments-in-video-with-natural-language

- [Activity-Captions]: https://github.com/ranjaykrishna/densevid_eval

- [Charades-STA]: https://github.com/jiyanggao/TALL

- [QVHighlights]: https://github.com/jayleicn/moment_detr

Then use ` reorganize_data.ipynb` to reorganize dataset in the tree datasets.

**Referring Video Object Segmentation**

- [MeViS]: https://github.com/henghuiding/MeViS

- [Ref-YT-VOS]: https://github.com/skynbe/Refer-Youtube-VOS

- [Ref-DAVIS17]: https://github.com/davisvideochallenge/davis2017-evaluation

Please refer 

[VISA]: https://github.com/cilinyan/VISA?tab=readme-ov-file#1-training-data-preparation

data prepare code. Our dataset code will process these styles of YouTube automatically.

## Training and Validation

Our config files are in ` config\`. Please determine their hyperparameters.

### Stage1

(Optional,  faster than video) Extracting Moment Retrieval dataset features 

```shell
sh run_stage1_extract.sh
```

Fine-tuning TCS

```shell
sh run_stage1.sh
```

Evaluating TCS

```shell
sh run_stage1_eval.sh
python mllm/tools/eval_mr_result_file.py
python mllm/tools/qv_submission_formatting.py
```

Predicting TCS with RVOS datasets

```shell
sh run_stage1_predict_rvos.sh
```

### Stage2

Fine-tuning DTOS

```shell
sh run_stage2.sh
```

Evaluating DTOS

```shell
sh run_stage2_eval.sh
python mllm/tools/auto_pack_and_transfer_files.py
```

## Performance

![RVOS_comp](assert\RVOS_comp.png)

![MR_comp](assert\MR_comp.png)

## Visualizations

![visualization](assert\visualization.png)

![more_visualizations](assert\more_visualizations.png)

## Acknowledgement

This work is built upon the 

[VILA]: https://github.com/NVlabs/VILA

, 

[SAM2]: https://github.com/facebookresearch/sam2

, 

[Next-Chat]: https://github.com/NExT-ChatV/NExT-Chat

, 

[SAM]: https://github.com/facebookresearch/segment-anything

, 

[Moment-DETR]: https://github.com/jayleicn/moment_detr

, 

[DETR]: https://github.com/facebookresearch/detr

, 

[LLaVA]: https://github.com/haotian-liu/LLaVA

and 

[MeViS]: https://github.com/henghuiding/MeViS

.