import os
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class MaskSaver:
    def __init__(self, output_dir, dataset_name, dataset_type, video_name, exp_id, max_workers=4):
        self.output_dir = output_dir
        self.exp_id = exp_id
        self.video_name = video_name
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        # if dataset_type == "val":
        #     self.dataset_type = "valid_u"
        # elif dataset_type == "test":
        #     self.dataset_type = "valid"
        if self.dataset_name == 'youtubervos':
            self.save_folder = os.path.join(output_dir, dataset_name, self.dataset_type, 'Annotations', video_name, exp_id)
        else:
            self.save_folder = os.path.join(output_dir, dataset_name, self.dataset_type, video_name, exp_id)
        os.makedirs(self.save_folder, exist_ok=True)
        self.lock = threading.Lock()
        self.max_workers = max_workers

    def save_mask_as_png(self, fn, mask):
        with self.lock:  # 确保文件名生成的线程安全
            file_name = f"{fn}.png"
        img_path = os.path.join(self.save_folder, file_name)
        
        # 将 mask 转换为 PIL 图像并保存
        img = Image.fromarray((mask * 255).astype('uint8'))
        img.save(img_path)

    def save_all_masks(self, pred_masks, frame_names):
        pred_masks = pred_masks.to(device='cpu').numpy()  # 将 pred_masks 转换为 numpy 数组
        assert len(pred_masks) == len(frame_names), "Number of masks and frame names should be the same."
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.save_mask_as_png, fn, mask) for fn, mask in zip(frame_names, pred_masks)]
            for future in as_completed(futures):
                future.result()  # 等待所有任务完成

# 使用示例
if __name__ == "__main__":
    # 假设 pred_masks 已经处理完毕并转换为 numpy 数组
    pred_masks = torch.rand([10, 100, 100])  # 示例数据
    
    output_dir = "./output"
    exp_id = "xdxd"
    video_name = "example_video_name"
    
    saver = MaskSaver(output_dir, exp_id, video_name)
    saver.save_all_masks(pred_masks)