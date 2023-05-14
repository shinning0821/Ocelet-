from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class ConsepDataset_Fast(BaseDataset):

    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=2, augmul=10, patch_size=None, obj_option="Objs", bkg_option="Bkgs"):
        super(
            ConsepDataset_Fast,
            self).__init__(data_dir, transform, option,
                 random_seed, n_classes, augmul, patch_size, obj_option, bkg_option)

    def __getitem__(self, ind):
        real_ind = ind % self.files_no 
        if self.transform:
            file_list = self.get_train_names(real_ind)
            image, obj, bkg, mask, region, points = self.random_read_subregion(file_list, random_seed=self.random_seeds[ind])

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

            mask = np.clip(mask,0,1)
            obj = obj / 255
            
            for k, v in points.items():
                counts += len(v)
                if len(v) > 0:
                    point_label[v[:, 0], v[:, 1]] = int(k)

            return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                    'points': torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'gt': torch.FloatTensor(mask),
                    'region': torch.FloatTensor(region),
                    'counts': counts,
                    'meta': {'index': ind}}
        else:
            file_name = self.files_name
            file_name.sort()
            file_name = file_name[real_ind].split('.')[0]

            image = sio.imread(
                 os.path.join(self.data_dir, self.option, "Images", file_name+".png"))[..., :3]
            mask = sio.imread(
                os.path.join(self.data_dir, self.option, "GTs", file_name+".png"))
            points = hu.load_json(
                os.path.join(self.data_dir, self.option, "Pts", file_name+".json"))
            counts = len(points)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0,
                    'counts': counts,
                    'meta': {'index': ind},
                    'gt': torch.FloatTensor(mask)}

    def get_train_names(self, number):

        file_name = self.files_name
        file_name.sort()
        file_name = file_name[number].split('.')[0]
        return os.path.join(self.data_dir, self.option, "Images", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "Objs", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "Bkgs", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "GTs", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "Regions", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "Pts", file_name+".json")


    def random_read_subregion(self, file_list, random_seed=False):
        if random_seed:
            np.random.seed(random_seed)
        random_state = np.random.random(size=(2,))
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file_list[0])
        file_reader.ReadImageInformation()
        image_size = file_reader.GetSize()
        
        extractindex = [int((img_dim-self.patch_size)*random_) for img_dim, random_ in zip(image_size, random_state)]
        file_reader.SetExtractIndex(extractindex)
        file_reader.SetExtractSize([self.patch_size, self.patch_size])

        return_item = [sitk.GetArrayFromImage(file_reader.Execute())[..., :3]]

        for file in file_list[1:-1]:
            file_reader.SetFileName(file)
            return_item.append(sitk.GetArrayFromImage(file_reader.Execute()))

        points_crop = dict()

        points = hu.load_json(file_list[-1])
        temp = []
        for i in range(len(points["1"][0])):
            temp.append((points["1"][0][i],points["1"][1][i]))
        points["1"] = temp

        for k, v in points.items():
            if len(v) == 0:
                points_crop[k] = v
            else:
                v = np.array(v)
                ind = np.logical_and(np.logical_and((v[:, 0]-extractindex[0]) >= 0,
                                                    (v[:, 0] < extractindex[0] + self.patch_size)),
                                     np.logical_and((v[:, 1]-extractindex[1]) >= 0,
                                                    (v[:, 1] < extractindex[1] + self.patch_size)))
                points_crop[k] = v[ind, :] - np.array(extractindex)[None]
        return_item.append(points_crop)
        return return_item

