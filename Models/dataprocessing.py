import numpy as np
import os
import torch
from PIL import Image
import cv2

import img_utils



class DatasetProcessing():
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        #print("len_img_filename:",len(self.img_filename))

        self.images = []
        with open(img_filepath, 'r') as f:
            self.images = f.readlines()
        #print("self.images:",len(self.images))
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.float32)

        self.label = labels
        #print("self.label:",len(self.label))
    def __getitem__(self, index):
        #print("index:",index)
        #index = index +1
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        #img = (np.float32(img), cv2.COLOR_RGB2GRAY)
        if self.transform is not None:
            
            img = self.transform(img)
            """img *= 255
            img /= 127.5 
            img -= 1"""
            

        label = torch.from_numpy(self.label[index])
        
        return img, label

    def __len__(self):
        return len(self.images) 