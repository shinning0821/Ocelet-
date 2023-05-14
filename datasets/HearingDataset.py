from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class HearingDataset(BaseDataset):
    
    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=2, augmul=1, obj_option=None):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir, option, "Images", "*.png")))
        self.obj_option = obj_option

        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul*self.files_no*100,
                                                  (self.augmul*self.files_no,))
        self.files_name = os.listdir(os.path.join(self.data_dir, self.option, "Images"))


    def __getitem__(self, ind):
        real_ind = ind % self.files_no 
        file_name = self.files_name
        file_name.sort()
        file_name = file_name[real_ind].split('.')[0]
        image = sio.imread(os.path.join(self.data_dir, self.option, "Images", file_name+".png"))[..., :3]
        mask = sio.imread(os.path.join(self.data_dir, self.option, "GTs", file_name+".png"))
        
        if self.transform:
            random_seed = self.random_seeds[ind]
            random.seed(random_seed)
            transformed = self.transform(image=image,
                                         mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
            mask = np.clip(mask,0,3)

        return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                'gt': torch.FloatTensor(mask),
                'meta': {'index': ind}}
    