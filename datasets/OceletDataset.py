from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk
from . import BaseDataset

class OceletDataset(BaseDataset):

    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=1, augmul=2, obj_option=None):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir,'annotations', option, "masks", "*.png")))
        self.obj_option = obj_option

        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul*self.files_no*100,
                                                  (self.augmul*self.files_no,))
        self.files_names = os.listdir(os.path.join(self.data_dir,'annotations', self.option, "masks"))
        self.pair = hu.load_json(os.path.join(self.data_dir,'metadata.json'))['sample_pairs']


    def __getitem__(self, ind):
        real_ind = ind % self.files_no 
        file_names = self.files_names
        file_names.sort()
        file_name = file_names[real_ind].split('.')[0]

        image = sio.imread(os.path.join(self.data_dir, 'images',self.option, "cell", file_name+".jpg"))[..., :3]
        mask = sio.imread(os.path.join(self.data_dir,'annotations', self.option, "masks", file_name+".png"))
        heatmap = sio.imread(os.path.join(self.data_dir,'annotations', self.option, "heatmap", file_name+".jpg"))
        tissue_img = sio.imread(os.path.join(self.data_dir, 'images',self.option, "tissue", file_name+".jpg"))[..., :3]
        tissue_mask = sio.imread(os.path.join(self.data_dir,'annotations', self.option, "tissue", file_name+".png"))

        offset_x = self.pair[file_name]["patch_x_offset"] * tissue_mask.shape[0]
        offset_y = self.pair[file_name]["patch_y_offset"] * tissue_mask.shape[0]
        # pair_map = np.zeros((tissue_mask.shape))
        # pair_map[int(offset_y-128):int(offset_y+128),int(offset_x-128):int(offset_x+128)] = 1
        loc = [(offset_x,offset_y)]

        if self.transform:
            random_seed = self.random_seeds[ind]
            random.seed(random_seed)
            transformed = self.transform(image=image,
                                            mask=mask,
                                            mask0=heatmap,
                                            tissue_img = tissue_img,
                                            tissue_mask = tissue_mask,
                                        #  pair_map = pair_map,
                                            keypoints = loc)
            image = transformed["image"]
            mask = transformed["mask"]
            heatmap = transformed['mask0']
            tissue_img = transformed['tissue_img']
            tissue_mask = transformed['tissue_mask']
            # pair_map = transformed['pair_map']
            loc = np.array(transformed["keypoints"][0])
        mask = np.clip(mask,0,2)

        return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                'heatmap': torch.FloatTensor(heatmap)/255.0,
                'gt': torch.FloatTensor(mask),
                'tissue_img': torch.FloatTensor(tissue_img.transpose(2, 0, 1))/255.0,
                'tissue_mask': torch.FloatTensor(tissue_mask),
                # 'pair_map': torch.FloatTensor(pair_map),
                'roi_loc': torch.IntTensor(loc),
                'meta': {'index': ind}}


class OceletDataset_Fast(BaseDataset):

    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=1, augmul=100, patch_size=None, obj_option=None, bkg_option=None):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir,'annotations', option, "masks", "*.png")))
        self.patch_size = patch_size
        self.obj_option = obj_option
        
        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul*self.files_no*100,
                                                  (self.augmul*self.files_no,))
        self.files_names = os.listdir(os.path.join(self.data_dir,'annotations', self.option, "masks"))

    def __getitem__(self, ind):
        real_ind = ind % self.files_no 
        if self.transform:
            file_list = self.get_train_names(real_ind)
            image, mask, heatmap = self.random_read_subregion(file_list, random_seed=self.random_seeds[ind])

            random_seed = self.random_seeds[ind]

            random.seed(random_seed)

            transformed = self.transform(image=image,                                         
                                         mask=mask,
                                         mask0=heatmap)
            image = transformed["image"]
            mask = transformed["mask"]
            heatmap = transformed['mask0']
            mask = np.clip(mask,0,2)
            
            # mask = np.clip(mask,0,3)

        else:
            file_names = self.files_names
            file_names.sort()
            file_name = file_names[real_ind].split('.')[0]

            image = sio.imread(os.path.join(self.data_dir, 'images',self.option, "cell", file_name+".jpg"))[..., :3]
            mask = sio.imread(os.path.join(self.data_dir,'annotations', self.option, "masks", file_name+".jpg"))
            heatmap = sio.imread(os.path.join(self.data_dir,'annotations', self.option, "heatmap", file_name+".jpg"))
            
        return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                'heatmap': torch.FloatTensor(heatmap)/255.0,
                'gt': torch.FloatTensor(mask),
                'meta': {'index': ind}}

    def get_train_names(self, number):

        file_names = self.files_names
        file_names.sort()
        file_name = file_names[number].split('.')[0]
        return os.path.join(self.data_dir, 'images',self.option, "cell", file_name+".jpg"), \
               os.path.join(self.data_dir,'annotations', self.option, "masks", file_name+".jpg"),\
               os.path.join(self.data_dir,'annotations', self.option, "heatmap", file_name+".jpg")


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
