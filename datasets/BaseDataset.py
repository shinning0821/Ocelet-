from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk

class BaseDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=1, augmul=1, patch_size=None, obj_option="Objs", bkg_option="Bkgs"):
        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir, option, "Images", "*.png")))
        self.obj_option = obj_option
        self.patch_size = patch_size
        self.bkg_option = bkg_option
        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul * self.files_no * 100,
                                                  (self.augmul * self.files_no,))
        self.files_names = os.listdir(os.path.join(self.data_dir, self.option, "Images"))

    def __getitem__(self, ind):
        raise Exception("Not Implemented")


    def __len__(self):
        if self.option is "train":
            return self.augmul*self.files_no
        else:
            return self.files_no