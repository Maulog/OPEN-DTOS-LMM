import copy
from typing import List, Union, Dict
import os
import re, json

import gradio as gr
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image

import PIL.Image
import torch
from torch import nn
import numpy as np
import torchvision.transforms.functional as F
import transformers
from transformers import AutoConfig
from transformers import StoppingCriteria
from transformers import PreTrainedTokenizer

from matplotlib import pyplot as plt

from mllm.config.constants import IMAGE_TOKEN_INDEX
# from mllm.models.dtos.dtos_base import DtosLmmForCausalLM
# from mllm.dataset.utils import expand2square

def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor):
    ids = copy.deepcopy(ids)  # do not modify origin preds and targets
    ids[ids < 0] = tokenizer.pad_token_id
    return ids


def decode_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor) -> Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer, ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res


def show(imgs: Union[torch.Tensor, List[Union[torch.Tensor, PIL.Image.Image]]]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def draw_bounding_boxes(
        image: Union[torch.Tensor, PIL.Image.Image],
        boxes: Union[torch.Tensor, List, np.ndarray],
        **kwargs,
):
    if isinstance(image, PIL.Image.Image):
        from torchvision.transforms import PILToTensor
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
    return _draw_bounding_boxes(image, boxes, **kwargs)


def smart_tokenizer_and_partial_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedTokenizer,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        # 用0初始化是因为不会使用到新增的token作为输出  
        input_embeddings[-num_new_tokens:] = torch.zeros_like(input_embeddings_avg, 
                                                              device=input_embeddings_avg.device,
                                                              dtype=input_embeddings_avg.dtype)
        output_embeddings[-num_new_tokens:] = torch.zeros_like(output_embeddings_avg, 
                                                               device=output_embeddings.device,
                                                               dtype=output_embeddings.dtype)
        
        input_init_weight = input_embeddings_avg.repeat(num_new_tokens, 1)
        # output_init_weight = output_embeddings_avg.repeat(num_new_tokens, 1)
        
        if not hasattr(model, "sam_loss"): # dtos_loc
            model.build_rec_token_projector(num_new_tokens, len(tokenizer), model.config.hidden_size, 
                                            init_weight=input_init_weight)
            # model.build_rec_token_classifier(model.config.hidden_size)
            
        else: # dtos_seg
            model.build_seg_token_projector(num_new_tokens, len(tokenizer), model.config.hidden_size, 
                                            init_weight=input_init_weight)
        
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
def smart_resize_dtos_loc(model, tokenizer): 
    # 该函数用于扩展输入层，并且构建特殊token的输出层，之后在加载参数的部分直接覆盖
    model.resize_token_embeddings(len(tokenizer))
    
    model.rec_token_projector = nn.Embedding(len(tokenizer), model.config.hidden_size) # 直接替换模型前向的投影层
    
    # output_embeddings = model.get_output_embeddings().weight.data
    # output_embeddings[-num_new_tokens:] = torch.zeros((1, model.config.hidden_size), 
    #                                                 device=output_embeddings.device,
    #                                                 dtype=output_embeddings.dtype)
    
    # model.build_rec_token_classifier(model.config.hidden_size)
    
def smart_resize_dtos_seg(model, tokenizer): 
    # 该函数用于扩展输入层，并且构建特殊token的输出层，之后在加载参数的部分直接覆盖
    model.resize_token_embeddings(len(tokenizer))
    
    model.seg_token_projector = nn.Embedding(len(tokenizer), model.config.hidden_size) # 直接替换模型前向的投影层
    
def load_state_dict_with_warning(model, state_dict):
    own_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    
    for name, param in state_dict.items():
        full_name = name
        if full_name not in own_state:
            unexpected_keys.append(full_name)
            print(f"Warning: Unexpected key '{full_name}' found in provided state_dict. Ignoring.")
        elif param.shape != own_state[full_name].shape:
            print(f"Warning: Mismatch in shape for key '{full_name}'. Expected {own_state[full_name].shape}, got {param.shape}. Ignoring.")
            unexpected_keys.append(full_name)
        else:
            own_state[full_name].copy_(param)
            
    # for name in own_state:
    #     if name not in state_dict:
    #         missing_keys.append(name)
    #         print(f"Warning: Missing key '{name}' in provided state_dict.")
            
    if len(unexpected_keys) > 0 or len(missing_keys) > 0:
        print('\n############################')
        print("Summary:")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {', '.join(unexpected_keys)}")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {', '.join(missing_keys)}")
        print('############################')
    else:
        print('\n#################################')
        print("All keys matched successfully.")
        print('#################################')
    

def is_mm_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    architectures = config.architectures
    for architecture in architectures:
        if "llava" in architecture.lower():
            return True
    return False



# 通过字符串返回对应的torch类型
def TorchDtypeMapper(dtype_str: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_str)
    except AttributeError as e:
        valid_dtypes = [dtype.name for dtype in torch.dtypes.all_types]
        raise AttributeError(
            f"Unknown PyTorch dtype string: {dtype_str}. Valid options include {valid_dtypes}"
        ) from e
            

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def disable_torch_init(): 
    # 函数通过将这两个类的 reset_parameters 方法替换为一个空操作（即 lambda self: None），禁用默认初始化
    # 使得在实例化时不进行任何初始化动作，从而提升了模型创建的速度。但请注意，这样做意味着开发者需要自己负责后续模型参数的正确初始化
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



class KeywordsStoppingCriteria(StoppingCriteria): 
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids)) # 将当前的停止token的ids存入列表中
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len) # 根据生成的文本长度和最大停止token长度，取最小值
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids: # 遍历停止token的ids
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all(): # 如果生成的字符id中包含停止字符id，则返回True
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs: # 如果生成的字符中包含停止字符，则返回True
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)





class ImageBoxState:
    def __init__(self, draw_size=512):
        if isinstance(draw_size, (float, int)):
            draw_size = (draw_size, draw_size)
        assert len(draw_size) == 2
        self.size = draw_size
        self.height, self.width = self.size[0], self.size[1]
        self.reset_state()
        self.cnt = 0

    # noinspection PyAttributeOutsideInit
    def reset_state(self):
        self.image = None
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def reset_masks(self):
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def update_image(self, image):
        if image != self.image:
            # self.reset_state()
            self.image = image

    def update_mask(self, mask):
        if len(self.masks) == 0:
            last_mask = np.zeros_like(mask)
        else:
            last_mask = self.masks[-1]

        if type(mask) == np.ndarray and mask.size > 1:
            diff_mask = mask - last_mask
        else:
            diff_mask = np.zeros([])

        # clear all of the strokes
        if mask.sum() == 0:
            self.reset_masks()
            return

        if (mask.astype(np.float32) - last_mask.astype(np.float32)).sum()<0:
            self.boxes.pop()
            self.masks.pop()
            return

        if diff_mask.sum() > 0:
            # noinspection PyArgumentList
            x1x2 = np.where(diff_mask.max(0) != 0)[0]
            # noinspection PyArgumentList
            y1y2 = np.where(diff_mask.max(1) != 0)[0]
            y1, y2 = y1y2.min(), y1y2.max()
            x1, x2 = x1x2.min(), x1x2.max()
            if (x2 - x1 > 5) and (y2 - y1 > 5):
                self.masks.append(mask.copy())
                self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def update_box(self, box):
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def to_model(self):
        pass
        # if self.image is None:
        #     return {}
        # image = expand2square(self.image)
        # boxes = [box_xyxy_expand2square(box, w=self.image.width, h=self.image.height) for box in self.boxes]
        # return {'image': image, 'boxes': boxes}

    def draw_boxes(self):
        assert self.image is not None
        grounding_texts = [f'{bid}' for bid in range(len(self.boxes))]
        def _draw(img, _boxes, texts):
            assert img is not None
            colors = ["red", "blue", "green", "olive", "orange", "brown", "cyan", "purple"]
            _img_draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'assets/DejaVuSansMono.ttf'), size=18)
            for bid, box in enumerate(_boxes):
                _img_draw.rectangle((box[0], box[1], box[2], box[3]), outline=colors[bid % len(colors)], width=4)
                anno_text = texts[bid]
                _img_draw.rectangle((box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]),
                                    outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
                _img_draw.text((box[0] + int(font.size * 0.2), box[3] - int(font.size * 1.2)), anno_text, font=font, fill=(255, 255, 255))
            return img

        out_draw = _draw(self.image, self.boxes, grounding_texts)
        return out_draw


def bbox_draw(sketch_pad: dict, state: dict):
    def binarize(x):
        return (x != 0).astype('uint8') * 255
    image = sketch_pad['image']
    image = open_image(image)
    # global count
    # count += 1
    # np.save( f"{count}.npy", sketch_pad['mask'])
    mask = sketch_pad['mask'].sum(-1) if sketch_pad['mask'].ndim == 3 else sketch_pad['mask']
    mask = binarize(mask)
    ibs = state["ibs"]
    ibs.update_image(image)
    ibs.update_mask(mask)
    out_draw = ibs.draw_boxes()
    return out_draw, state


def open_image(image):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    elif type(image) == str:
        image = Image.open(image).convert("RGB")
    return image


def parse_boxes(text):
    def is_valid(lst):
        return all([(type(x) == int) and (x >= 0) for x in lst]) and len(lst)>0
    text = text.replace(",]", "]")
    pat = re.compile(r"\[.*?\]")
    matched_boxes = pat.findall(text)
    ret_boxes = []
    to_sub_strs = []
    for box_str in matched_boxes:
        try:
            box_seq = json.loads(box_str)
            if is_valid(box_seq):
                ret_boxes.append(box_seq)
                text = text.replace(box_str, "{}")
                # to_sub_strs.append(" ".join(["<at> <boxes>"]*len(box_seq)))
                to_sub_strs.append("<at> <boxes>")
        except Exception as e:
            pass
    text = text.format(*to_sub_strs)
    return text, ret_boxes