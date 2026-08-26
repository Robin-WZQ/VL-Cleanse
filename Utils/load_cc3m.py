import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import json

# 设置设备
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# 数据路径
data_path = "/mnt/sdb1/wangzhongqi/Dataset/CC3M"
output_file = "/mnt/sdb1/wangzhongqi/Dataset/CC3M_backdoor_list.json"  # 保存清洗后的样本列表

# 获取所有tar文件
train_tars = [os.path.join(data_path, f"cc3m-train-{i:04d}.tar") for i in range(576)]
val_tars = [os.path.join(data_path, f"cc3m-validation-{i:04d}.tar") for i in range(16)]
all_tars = train_tars + val_tars

backdoor_samples = []

for tar_path in tqdm(all_tars[:1], desc="Processing tars"):
    if not os.path.exists(tar_path):
        continue
    
    with tarfile.open(tar_path, 'r') as tar:
        # 获取所有成员，按名称排序以配对
        members = tar.getmembers()
        # 假设文件名匹配，如 00000000.jpg 和 00000000.txt
        image_members = {m.name: m for m in members if m.name.endswith('.jpg')}
        text_members = {m.name: m for m in members if m.name.endswith('.txt')}
        i=0
        for img_name, img_member in tqdm(image_members.items(), desc="Processing images"):
            txt_name = img_name.replace('.jpg', '.txt')
            if txt_name in text_members:

                # 读取图像
                img_file = tar.extractfile(img_member)
                image_data = img_file.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                # 保存图像
                
                image.save(f"/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Data/CC3M_data/{img_name.split('/')[-1]}")
                i+=1
                
            if i>=6000:
                break
                    
                    