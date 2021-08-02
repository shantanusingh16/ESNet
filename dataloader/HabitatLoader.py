import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from utils import preprocess 
from skimage import transform, io

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class HabitatDataset(data.Dataset):
    def __init__(self, split_path, training, loader=default_loader, dploader= disparity_loader):
 
        self.split_path = split_path
        with open(split_path, 'r') as f:
            self.left_filepaths = f.read().splitlines()
        self.loader = loader
        self.dploader = dploader
        self.training = training

        self.img_size = (480, 640)
        self.scale_size = (384, 512)
        self.baseline = 0.2
        self.focal_length = 320
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __getitem__(self, index):
        left = self.left_filepaths[index]
        right = self.left_filepaths[index].replace('left_rgb', 'right_rgb')
        depth_L = self.left_filepaths[index].replace('left_rgb', 'left_depth').replace('jpg', 'png')

        img_names = [left, right, depth_L]

        left_img = self.loader(left)
        right_img = self.loader(right)
        depth_arr = self.dploader(depth_L)

        if self.training:  
           w, h = left_img.size
           th, tw = 384, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(depth_arr,dtype=np.float32)/6553.5
           dataL = (self.baseline * self.focal_length)/dataL
           dataL = np.nan_to_num(dataL, posinf=0, neginf=0)
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img = processed(left_img)
           right_img = processed(right_img)

        #    return left_img, right_img, dataL
        else:
           w, h = left_img.size
           
           dataL = np.ascontiguousarray(depth_arr,dtype=np.float32)/6553.5
           dataL = (self.baseline * self.focal_length)/dataL
           dataL = np.nan_to_num(dataL, posinf=0, neginf=0)

           processed = preprocess.scale_transform(scale_size=self.get_scale_size()) # preprocess.get_transform(augment=False)
           left_img = processed(left_img)
           right_img = processed(right_img)

        #    return left_img, right_img, dataL

        gt_disp = dataL[np.newaxis, :]
        gt_disp = torch.from_numpy(gt_disp.copy()).float()

        sample = {  'img_left': left_img, 
                    'img_right': right_img,
                    'gt_disp': gt_disp,
                    'img_names': img_names
                 }

        return sample

    def __len__(self):
        return len(self.left_filepaths)


if __name__ == '__main__':
    dataset = HabitatDataset('/home/shantanu.singh/ESNet/lists/habitat_train_depth.list', True)
    sample = dataset[0]