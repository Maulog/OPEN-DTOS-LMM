from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from torch.nn import functional as F



class ImageGrid:
    def __init__(self) -> None:
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
    def get_single_image_grid(self, img_list, image_rows, image_cols):
        '''
        input: x = [pil, pil, pil, pil, ...] 
        output: pil
        '''
        tensor_img_list = [self.to_tensor(pil_img) for pil_img in img_list]
        c, h, w = tensor_img_list[0].shape
        
        assert len(tensor_img_list) == image_rows*image_cols, "batch size not match grid_size"
        # 按行拼接
        imgs = [torch.cat(tensor_img_list[row*image_cols:(row+1)*image_cols], dim=-1) for row in range(image_rows)]
        imgs = torch.cat(imgs, dim=-2)
        
        imgs = F.interpolate(imgs.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        imgs = self.to_pil(imgs.squeeze(0))
        return imgs
    
    def get_single_image_grid_col(self, img_list, image_rows, image_cols):
        '''
        input: x = [pil, pil, pil, pil, ...] 
        output: pil
        '''
        tensor_img_list = [self.to_tensor(pil_img) for pil_img in img_list]
        c, h, w = tensor_img_list[0].shape
        
        assert len(tensor_img_list) == image_rows*image_cols, "batch size not match grid_size"
        # 按行拼接
        imgs = [torch.cat(tensor_img_list[col:image_rows*image_cols:image_cols], dim=-2) for col in range(image_cols)]
        imgs = torch.cat(imgs, dim=-1)
        
        imgs = F.interpolate(imgs.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        imgs = self.to_pil(imgs.squeeze(0))
        return imgs
    
    def get_video_and_image_ig_id(self, indices, image_num, rough_img_num_sub_interval):
        '''
        input: indices = [0, 1, 2, 3, ...]
        output: (video_indices, image_indices)
        # 这个函数只操作索引，通常详细帧等于粗略帧，拼接时为<image><video>这样拼接
        '''
        image_step_len = len(indices) // image_num
        image_indices = np.arange(0, len(indices), image_step_len, dtype=int)[:image_num] # 开头帧为详细帧
        
        tmp_img_indices = image_indices.copy()
        tmp_img_indices = np.append(tmp_img_indices, indices[-1]) # 增加结束元素
        video_interval = [(tmp_img_indices[i], tmp_img_indices[i+1]) for i in range(len(tmp_img_indices)-1)]
        video_indices = []
        for start, end in video_interval:
            sub_interval = np.linspace(start, end, rough_img_num_sub_interval+2, dtype=int) # 两头为详细帧，不要
            sub_interval = sub_interval[1:-1]
            video_indices.append(sub_interval)
            # video_indices = np.append(video_indices, sub_interval)
        return video_indices, image_indices
    
    def get_video_ig_id(self, indices, video_num, rough_img_num_sub_interval):
        '''
        input: indices = [0, 1, 2, 3, ...]
        output: video_indices
        # 这个函数只操作索引，video_num是最终采集的视频的总数
        # rough_img_num_sub_interval是一个子片段采样数，拼接时为<video><video>这样拼接
        '''
        assert len(indices) >= video_num * rough_img_num_sub_interval, "indices is not enough"
        all_sample_num = video_num * rough_img_num_sub_interval
        sample_indices = np.linspace(0, len(indices)-1, all_sample_num, dtype=int)
        # 重新按rough_img_num_sub_interval整理分组
        video_indices = []
        for i in range(video_num):
            video_indices.append(sample_indices[i*rough_img_num_sub_interval:(i+1)*rough_img_num_sub_interval])
        return video_indices

    
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # original_img = Image.open('/home/tianjirui/DTOS-LMM/demo/traffic_light.jpg')
    # img_list = [original_img for _ in range(9)]
    # img_grid = ImageGrid(3, 3)
    # img = img_grid.get_single_image_grid(img_list)
    # plt.imshow(img)
    # plt.show()
    
    indices = np.arange(0, 100, 1)
    img_grid = ImageGrid()
    vid_idx, img_idx = img_grid.get_video_and_image_ig_id(indices, image_num=10, rough_img_num_sub_interval=4)
    print(vid_idx)
    print(img_idx)