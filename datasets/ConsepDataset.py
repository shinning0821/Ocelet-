from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class ConsepDataset(BaseDataset):
    
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
        if self.obj_option == "Gauss":
            obj = sio.imread(os.path.join(self.data_dir, self.option, "GaussObj", file_name + ".png"))
        else:
            obj = sio.imread(os.path.join(self.data_dir, self.option, "Objs", file_name+".png"))
        bkg = sio.imread(os.path.join(self.data_dir, self.option, "Bkgs", file_name+".png"))
        mask = sio.imread(os.path.join(self.data_dir, self.option, "GTs", file_name+".png"))
        region = sio.imread(os.path.join(self.data_dir, self.option, "Regions", file_name+".png"))
        points = hu.load_json(os.path.join(self.data_dir, self.option, "Pts", file_name+".json"))
        
        temp = []
        for i in range(len(points["1"][0])):
            temp.append((points["1"][0][i],points["1"][1][i]))
        points["1"] = temp

        if self.transform:
            random_seed = self.random_seeds[ind]
            random.seed(random_seed)
            transformed = self.transform(image=image,
                                         keypoints=points["1"],
                                         mask=mask,
                                         mask0=bkg,
                                         mask1=obj,
                                         mask2=region)
            image = transformed["image"]
            points["1"] = np.array(transformed["keypoints"]).astype(int)
            mask = transformed["mask"]
            bkg = transformed["mask0"]
            obj = transformed["mask1"]
            region = transformed["mask2"]
            point_label = np.zeros_like(mask)
            counts = 0

            for k, v in points.items():
                counts += len(v)
                if len(v) > 0:
                    point_label[v[:, 0], v[:, 1]] = int(k)
            
            mask = np.clip(mask,0,1)
            obj = obj / 255 

            return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                    'points': torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'gt': torch.FloatTensor(mask),
                    'region': torch.FloatTensor(region),
                    'counts': counts,
                    'meta': {'index': ind}}
        else:
            counts = len(points)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                    'counts': counts,
                    'meta': {'index': ind},
                    'gt': torch.FloatTensor(mask)}
    