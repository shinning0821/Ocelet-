from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class HearingDataset_Fast(BaseDataset):

    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=2, augmul=100, patch_size=None, obj_option="Objs", bkg_option="Bkgs"):
        super(
            HearingDataset_Fast,
            self).__init__(data_dir, transform, option,
                 random_seed, n_classes, augmul, patch_size, obj_option, bkg_option)

    def __getitem__(self, ind):
        real_ind = ind % self.files_no 
        if self.transform:
            file_list = self.get_train_names(real_ind)
            image, mask = self.random_read_subregion(file_list, random_seed=self.random_seeds[ind])

            random_seed = self.random_seeds[ind]

            random.seed(random_seed)

            transformed = self.transform(image=image,                                         
                                         mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            mask = np.clip(mask,0,3)

        else:
            file_name = self.files_name
            file_name.sort()
            file_name = file_name[real_ind].split('.')[0]

            image = sio.imread(
                 os.path.join(self.data_dir, self.option, "Images", file_name+".png"))[..., :3]
            mask = sio.imread(
                os.path.join(self.data_dir, self.option, "GTs", file_name+".png"))
            
        return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                'gt': torch.FloatTensor(mask),
                'meta': {'index': ind}}

    def get_train_names(self, number):

        file_name = self.files_name
        file_name.sort()
        file_name = file_name[number].split('.')[0]
        return os.path.join(self.data_dir, self.option, "Images", file_name+".png"), \
               os.path.join(self.data_dir, self.option, "GTs", file_name+".png")


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

        for file in file_list[1:]:
            file_reader.SetFileName(file)
            return_item.append(sitk.GetArrayFromImage(file_reader.Execute()))

        return return_item

