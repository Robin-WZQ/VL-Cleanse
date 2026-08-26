import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Feature_Dataset(Dataset):
    def __init__(self,file_path,processor,sample_scale,mode,image_size):
        self.path = file_path
        self.processor = processor
        self.file_list = []
        self.image_size = image_size
        
        for file in os.listdir(file_path)[:sample_scale]:
            self.file_list.append(file)
        
        if mode == "train":
            self.file_list = self.file_list[:int(len(self.file_list)*0.80)]
        elif mode == "test":
            self.file_list = self.file_list[int(len(self.file_list)*0.80):]
        else:
            raise ValueError("mode should be 'train' or 'test'")
        
        self.transform = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        image_path = os.path.join(self.path,self.file_list[index])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), resample=3)
        
        image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.file_list)