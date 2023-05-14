# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,feature
from skimage.feature import peak_local_max as plm
import pandas as pd
import os

root_dir = '/data114_1/wzy/homework/code/eval/predict'
mask_dir = os.path.join(root_dir,'masks')
csv_dir = os.path.join(root_dir,'csv')

paths = os.listdir(mask_dir)
paths.sort()
for i,path in enumerate(paths):
    file_name = path.split('.')[0]
    img = cv2.imread(os.path.join(mask_dir,path))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distance = ndi.distance_transform_edt(image)
    coordinates = plm(distance)
    df = pd.DataFrame({'x':coordinates[:,1],'y':coordinates[:,0]})
    df['lb']=''
    for index,row in df.iterrows():
        df['lb'][index]=image[row['y']][row['x']]
    
    df.to_csv(os.path.join(csv_dir,file_name+'.csv'),index=False,header=0)
    
    
root_dir = '/data114_1/wzy/homework/code/eval/gt'
mask_dir = os.path.join(root_dir,'masks')
csv_dir = os.path.join(root_dir,'csv')

paths = os.listdir(mask_dir)
paths.sort()
for i,path in enumerate(paths):
    file_name = path.split('.')[0]
    img = cv2.imread(os.path.join(mask_dir,path))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distance = ndi.distance_transform_edt(image)
    coordinates = plm(distance)
    df = pd.DataFrame({'x':coordinates[:,1],'y':coordinates[:,0]})
    df['lb']=''
    for index,row in df.iterrows():
        df['lb'][index]=image[row['y']][row['x']]
    
    df.to_csv(os.path.join(csv_dir,file_name+'.csv'),index=False,header=0)
    
    
    
    