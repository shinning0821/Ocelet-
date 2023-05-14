from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class HEDataset(BaseDataset):
    
    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=7, augmul=10, obj_option=None):
        super(
            HEDataset,
            self).__init__(data_dir, transform, option,
                 random_seed, n_classes, augmul, obj_option)
                 
    def __getitem__(self, ind):
        real_ind = ind % self.files_no + 1
        if self.transform:
            file_list = self.get_train_names(real_ind)
            image, obj, bkg, mask, region, points = self.random_read_subregion(file_list, random_seed=self.random_seeds[ind])

            random_seed = self.random_seeds[ind]

            random.seed(random_seed)

            transformed = self.transform(image=image,
                                         keypoints=points["1"],
                                         keypoints0=points["2"],
                                         keypoints1=points["3"],
                                         keypoints2=points["4"],
                                         keypoints3=points["5"],
                                         keypoints4=points["6"],
                                         keypoints5=points["7"],
                                         mask=mask,
                                         mask0=bkg,
                                         mask1=obj,
                                         mask2=region)
            image = transformed["image"]
            points["1"] = np.array(transformed["keypoints"]).astype(int)
            points["2"] = np.array(transformed["keypoints0"]).astype(int)
            points["3"] = np.array(transformed["keypoints1"]).astype(int)
            points["4"] = np.array(transformed["keypoints2"]).astype(int)
            points["5"] = np.array(transformed["keypoints3"]).astype(int)
            points["6"] = np.array(transformed["keypoints4"]).astype(int)
            points["7"] = np.array(transformed["keypoints5"]).astype(int)
            mask = transformed["mask"]
            bkg = transformed["mask0"]
            obj = transformed["mask1"]
            region = transformed["mask2"]

            point_label = np.zeros_like(mask)
            counts = 0
            for k, v in points.items():
                counts += len(v)
                if len(v) > 0:
                    point_label[v[:, 1], v[:, 0]] = int(k)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0,
                    'points': torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'gt': torch.FloatTensor(mask),
                    'region': torch.FloatTensor(region),
                    'counts': counts,
                    'meta': {'index': ind}}
        else:
            image = sio.imread(
                os.path.join(self.data_dir, self.option, "Norms", self.option.lower() + '_' + str(real_ind) + ".png"))[
                    ..., :3]
            mask = sio.imread(
                os.path.join(self.data_dir, self.option, "GTs", self.option.lower() + '_' + str(real_ind) + ".tif"))
            points = hu.load_json(
                os.path.join(self.data_dir, self.option, "Pts", self.option.lower() + '_' + str(real_ind) + ".json"))
            counts = len(points)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0,
                    'counts': counts,
                    'meta': {'index': ind},
                    'gt': torch.FloatTensor(mask)}
