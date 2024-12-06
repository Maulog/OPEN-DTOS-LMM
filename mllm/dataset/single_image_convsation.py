import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type
import os
import random
import copy
import re
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import TrainingArguments

from .root import IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER
from llava.conversation import Conversation, get_conv_template
from ..utils import post_process_generate_ids
from mllm.dataset.utils.io import read_jsonl_file, read_json_file, load_video, read_txt_file
from mllm.config.constants import *



class SingleImageConvDatasetMixin: # 单张图片对话数据集(基础类)

    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='llama_3'),
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            transforms: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.transforms = transforms

    def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            has_image = 'image' in item and bool(item['image'])
            has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
            if has_target and has_image:
                target['width'], target['height'] = image.width, image.height

        # preprocess
        raw_conv = self.process_conv(raw_conv) # replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        raw_conv, image = self.process_conv_multimage(raw_conv, image) # 如果是多图组织成列表进行返回
        raw_conv, tar_boxes = self.process_target(raw_conv, target, multimage_mode=multimage_mode) # normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        conv = self.build_conv(raw_conv) # 返回Conversation对象
        if return_conv:
            # noinspection PyTypeChecker
            return conv
        text_dict = self.process_text(conv) # convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.self.tokenize_kwargs control something like padding/truncation behavior.
        image_dict = self.process_image(image) # convert Image.Image object to torch.Tensor

        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        ret_dict["loc_inputs"] = tar_boxes["all_boxes"]
        ret_dict["loc_targets"] = tar_boxes["gpt_boxes"]
        self._print_sample(ret_dict, raw_conv, conv)
        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict

    def __len__(self):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def process_conv_multimage(self, raw_conv, image): 
        # re-sort multi image
        if image is None:
            return raw_conv, image
        if not isinstance(image, (list, tuple)):
            return raw_conv, image
        image_seqs = []
        for conv in raw_conv:
            image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
        images = []
        for idx in image_seqs:
            images.append(image[idx])
        return raw_conv, images

    def get_raw_item(self, index) -> Dict[str, Any]:
        """
        return item format like this.
        item = {
            'image': # PIL.Image.Image,
            'target': {
                # xmin, ymin, xmax, ymax
                'boxes': [
                    [10, 10, 256, 265],  # dog1
                    [24, 18, 378, 768],  # dog2
                    [100, 310, 670, 653],  # man
                    [278, 320, 809, 673],  # rope
                ],
            }

            "conversations": [
                {
                    'from': 'human',
                    'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
                    'boxes_seq': [[0, 1], [2], ],
                },
                {
                    'from': 'gpt',
                    'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                             'So the man <boxes> is walking the dog <boxes>.'
                            'And the man <boxes> has no relationship with the right dog <boxes>',
                    'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
                }
            ]
        }
        # placeholder: <image> <boxes>
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def validate_raw_item(self, item):
        has_image = 'image' in item and bool(item['image'])
        has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
        has_target_boxes = 'boxes' in item['target'] if has_target else False
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # check image
        human_input_has_image_placeholder = any(
            sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if human_input_has_image_placeholder:
            assert has_image
        if has_image and (not human_input_has_image_placeholder):
            warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
        gpt_input_has_image_placeholder = any(
            sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        assert not gpt_input_has_image_placeholder

        # check target
        has_boxes_placeholder = any(
            BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if has_boxes_placeholder:
            assert has_target_boxes
        # not check box placeholder num this will be checked in format process

    def build_conv(self, source: List[Dict[str, Any]]) -> Conversation:
        conv = self.conv_template()
        role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        for sentence in source:
            role = role_map[sentence['from']]
            conv.append_message(role, sentence['value'])
        return conv

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)

    def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor)

    def _print_sample(self, ret_dict, raw_conv, conv):
        if not hasattr(self, '_printed_sample'):
            self._printed_sample = True
            post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
            print(f"=================== {self.mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
            if 'image' in ret_dict and ret_dict['image'] is not None:
                image = ret_dict['image']
                if isinstance(image, torch.Tensor):
                    print(f"            image: {image.shape}")
                elif isinstance(image, dict):
                    print(f"            image: {image.keys()}")
                elif isinstance(image, list) and len(image) > 0:
                    print(f"            image: {len(image)}, {type(image[0])}")
                else:
                    print(f"            image: {type(image)}")
            print("====================================================", flush=True)
            try:
                if self.training_args is not None:
                    _save_obj = {
                        'ret_dict': ret_dict,
                        'raw_conv': raw_conv,
                        'conv': conv.get_prompt(),
                    }
                    from pathlib import Path
                    output_dir = Path(self.training_args.output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    _local_rank = self.training_args.local_rank
                    _word_size = self.training_args.world_size
                    _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
                    print(f'saving some sample to {_file_path} for check.')
                    torch.save(_save_obj, _file_path)
            except Exception as e:
                warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


class SingleImageConvDataset(SingleImageConvDatasetMixin, Dataset): # 单张图片对话数据集
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self): # 用dataset_generator生成dataset
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator()

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str: # 打印时调用
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


from mllm.models.sam.transforms import ResizeAndPad
class SingleImageConvSegDataset(SingleImageConvDatasetMixin, Dataset):
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None
        self.sam_transform = ResizeAndPad(1024)

    def initialize_if_needed(self):
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator()

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        ret_w, ret_h = image.width, image.height
        ret_masks = target.get("masks", None)

        # sam transform
        sam_image, sam_masks, sam_hw = self.sam_transform(image, target.get("masks", None))

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            has_image = 'image' in item and bool(item['image'])
            has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
            if has_target and has_image:
                target['width'], target['height'] = image.width, image.height


        # preprocess
        raw_conv = self.process_conv(raw_conv)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, tar_boxes = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv)
        if return_conv:
            # noinspection PyTypeChecker
            return conv
        text_dict = self.process_text(conv)
        image_dict = self.process_image(image)

        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        ret_dict["loc_inputs"] = tar_boxes["all_boxes"]
        ret_dict["loc_targets"] = tar_boxes["gpt_boxes"]
        ret_dict["images_sam"] = sam_image
        ret_dict["masks_sam"] = sam_masks
        ret_dict["img_size"] = torch.tensor([ret_h, ret_w])
        ret_dict["unresized_masks"] = torch.tensor(ret_masks[0])
        self._print_sample(ret_dict, raw_conv, conv)
        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict
        


class SingleVideoLocConvDatasetMixin: # 用于加载mr数据集
    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='llama_3'), # 已经有copy
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            model_args: Dict[str, Any] = None,
            data_args: Dict[str, Any] = None,
            transforms: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.transforms = transforms
        
        self.data_augmentation = True

    def __getitem__(self, data_index, debug_mode=False, return_conv=False) -> Dict[str, Any]: 
            
            item = self.get_raw_item(data_index)
            
            if item['im_src'] is not None:
                item['im_src'] = self.process_image(item['im_src'])
            if item['vi_src'] is not None:
                item['vi_src'] = self.process_image(item['vi_src'])
            
            timestamps, norm_timestamps, clip_length, captions = self.process_target(item)
            item['captions'] = captions # 不能放最后，创建对话用到
            
            raw_conv, reverse_list = self.build_conv(item) # 通过标签生成对话，添加固定的前缀
            raw_conv = self.process_conv(raw_conv) # 将<image>替换为 <im_start> <im_patch>*256 <im_end>
            
            text_dict = self.process_text(raw_conv) # 处理对话，返回输入的input_ids和attention mask和label
            
            item["conversation"] = text_dict
            item["timestamps"] = timestamps
            item["norm_timestamps"] = norm_timestamps
            item["clip_length"] = clip_length
            item["reverse_list"] = reverse_list
            
            # item_new = copy.deepcopy(item)
            # del item
            return item

    def get_raw_item(self, index) -> Dict[str, Any]: 
        raise NotImplementedError
    
    
    def __len__(self):
        raise NotImplementedError
    

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)

    def process_target(self, item) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]: # 将特殊的token替换为实际的信息
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](item, self.mode, self.data_augmentation)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor, self.model_args)
    
    def build_conv(self, item):
        
        def joint_captions(captions_list):
            result = ''
            for s in captions_list:
                result += s + '\n'
            return result
        
        # captions = joint_captions(item['captions'])
        captions_num = len(item['captions'])
        
        frm_num_list = np.linspace(0, 100, self.data_args.n_frms, endpoint=False, dtype=int) # 不同帧数25，这里可以修改到config文件中(必须改！！！！)
        vid_num_list = [f'{i+1}' for i in range(100)]
        
        
        # frame_sentence_list = DEFAULT_IMAGE_TOKEN + '\n'
        frame_sentence = ''.join([f'{i}{DEFAULT_IMAGE_TOKEN}{DEFAULT_VIDEO_TOKEN}\n' for i in frm_num_list]) # {DEFAULT_VIDEO_TOKEN}
        
        
        video_and_frames = item['prefix'] + frame_sentence # item['prefix'] + frame_sentence # 记得改
        
        conv = self.conv_template()
        
        # prompt = video_and_frames + '\n' # item['prefix'] + '\n' + video_and_frames + '\n' 
        # rec_question_reverse = 'Can you describe the location of these moments in the video?\n' # 之后用，记得改prompt
        
        def rec_query(caption):
            return 'Can you locate this description in the video?\n' + caption + '\n'
        
        def rec_ans(timestamps): # len(timestamps) 这里之后可以尝试随机但是在[len(t),5]之间的数
            # prefix = 'After careful consideration, I would like to locate these moments in the video as follows:'
            # suffix = 'Above are my detailed answers based on my analysis of the video.'
            
            # # content = DEFAULT_MOMENT_TOKEN*len(timestamps)
            # # content = ''
            # # for i in range(5):
            # #     content += (f'{i}<rec>')
            # content = 'First moment occurs in <rec>\n \
            #            The second location is <rec>\n \
            #            The third moment happen in <rec>\n \
            #            Fourth clip show up <rec>\n \
            #            Fifth fragment appear to <rec>\n'
                       
            # return prefix + content + suffix # 改这
        
            prefix = 'Considered, I would pinpoint ten video moments as:'
            suffix = 'My analysis yields detailed answers above.'
            # content = 'First <rec> Second <rec> Third <rec> Fourth <rec> Fifth <rec> End.'
            content = ''
            for i in range(10):
                content += (f'{DEFAULT_MOMENT_TOKEN}')
            content += '. '
            return prefix + content + suffix # 改这
        
        def rec_query_reverse(timestamps): # 暂时没用到
            return f'Can you describe {DEFAULT_MOMENT_TOKEN*len(timestamps)} in the video?\n'       
        
        def rec_ans_reverse(caption):
            return f'The moment describes that {caption}'
        
        reverse_list = []
        for i in range(captions_num): # 这里加入反向（）改这
            # use_reverse = random.choice([True, False]) if self.mode == 'train' else False
            use_reverse = False
            if use_reverse:
                user_question = rec_query_reverse(item['timestamps'][i])
                system_answer = rec_ans_reverse(item['captions'][i])
            else: # 不用反向
                user_question = rec_query(item['captions'][i])
                system_answer = rec_ans(item['timestamps'][i])
            if i == 0:
                user_question = video_and_frames + user_question # 万恶之源啊
            conv.append_message(conv.roles[0], user_question)
            conv.append_message(conv.roles[1], system_answer)
            reverse_list.append(use_reverse)
        return conv, reverse_list

    



class SingleVideoLocConvDataset(SingleVideoLocConvDatasetMixin, Dataset): 
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self): # 延迟加载dataset
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator() # 用dataset_generator生成dataset

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str: # 打印时调用
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class SingleVideoSegConvDatasetMixin: # 用于加载rvos数据集
    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='llama_3'), # 已经有copy
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            model_args: Dict[str, Any] = None,
            data_args: Dict[str, Any] = None,
            transforms: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.transforms = transforms
        
        self.data_augmentation = True

    def __getitem__(self, data_index, debug_mode=False, return_conv=False) -> Dict[str, Any]: 
            
            item = self.get_raw_item(data_index)
            
            if item['im_src'] is not None:
                item['im_src'] = self.process_image(item['im_src'])
            
            norm_bbox, tgt_frm_idx = self.process_target(item)
            item["tgt_frm_idx"] = tgt_frm_idx
            
            raw_conv = self.build_conv(item) # 通过标签生成对话，添加固定的前缀
            raw_conv = self.process_conv(raw_conv) # 将<image>替换为 <im_start> <im_patch>*256 <im_end>
            
            text_dict = self.process_text(raw_conv) # 处理对话，返回输入的input_ids和attention mask和label
            
            item["conversation"] = text_dict
            item["norm_bbox"] = norm_bbox
            
            
            
            return item

    def get_raw_item(self, index) -> Dict[str, Any]: 
        raise NotImplementedError
    
    
    def __len__(self):
        raise NotImplementedError
    

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)

    def process_target(self, item) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]: # 将特殊的token替换为实际的信息
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](item)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor, self.model_args)
    
    def build_conv(self, item):
                
        # captions = joint_captions(item['captions'])
        captions_num = len(item['captions'])
        
        frm_num_list = np.linspace(0, 10, self.data_args.n_frms, endpoint=False, dtype=int) # 不同帧数25，这里可以修改到config文件中(必须改！！！！)
        vid_num_list = [f'{i+1}' for i in range(100)]
        
        frame_sentence_list = []
        for j, per_sample_num in enumerate(item['per_moment_sample_num']):
            moment_sentence = ''
            for i in range(per_sample_num):
                offset = sum(item['per_moment_sample_num'][:j])
                per_frm_sentence = f'<tgt_start><tgt_{offset+i+1}><image><tgt_{offset+i+1}><tgt_end>\n'
                moment_sentence += per_frm_sentence
                
            frame_sentence_list.append(f'{DEFAULT_MOM_START_TOKEN}\n{moment_sentence}{DEFAULT_MOM_END_TOKEN}\n')
        frame_sentence = ''.join(frame_sentence_list)
        
        video_and_frames = item['prefix'] + frame_sentence # 记得改
        
        conv = self.conv_template()
        
        def rec_query(tgt_frm_idx, caption):
            mapper = {0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth',
                      5: 'sixth', 6: 'seventh', 7: 'eighth', 8: 'ninth', 9: 'tenth'} # {mapper[tgt_frm_idx]}
            return f'Could you identify the locations depicted in <tgt_{tgt_frm_idx+1}> image?\n' + caption + '\n'
        
        def rec_ans(is_valid): # 单独解码每个点的置信度和坐标
            # 这里设置只对三帧进行解码预测
            prefix = 'Sure! The image shows the following locations: '
            suffix = 'These are the places depicted.' # 这里取消box的前缀
            if is_valid:
                content = f'{DEFAULT_BOX_TOKEN*7}. '
            else:
                content = 'No match for the target. '
            
            return prefix + content + suffix
        
        
        # reverse_list = []
        for i, idx in enumerate(item['tgt_frm_idx']):
            user_question = rec_query(idx, item['captions'][0])
            system_answer = rec_ans(item['valid'][idx].item())
            if i == 0:
                user_question = video_and_frames + user_question # 万恶之源啊
            conv.append_message(conv.roles[0], user_question)
            conv.append_message(conv.roles[1], system_answer)
            
            if i == 0 and self.mode in ['validation', 'test']: # 如果是val或者test则直接返回
                break
            
        return conv

    



class SingleVideoSegConvDataset(SingleVideoSegConvDatasetMixin, Dataset): 
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self): # 延迟加载dataset
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator() # 用dataset_generator生成dataset

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str: # 打印时调用
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    

__all__ = ['SingleImageConvDatasetMixin', 'SingleImageConvDataset', 'SingleVideoLocConvDatasetMixin',
           'SingleVideoSegConvDatasetMixin']
WRAPPER_DATASET = { # 新增视频数据集
    "conv": SingleImageConvDataset,
    "conv_loc": SingleVideoLocConvDataset,
    "conv_seg": SingleVideoSegConvDataset,
}
