#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:40:32 2021

@author: user
"""
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
import os

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,events_dir, image_size,transform = None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.events_dir = events_dir
        self.image_size = image_size

        self.ids = [splitext(file)[0] for file in sorted(listdir(imgs_dir))
                    if not file.startswith('.')]
        self.ids_m = [splitext(file)[0] for file in sorted(listdir(masks_dir))
                    if not file.startswith('.')]
        self.ids_e = [splitext(file)[0] for file in sorted(listdir(events_dir))
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.transform = transform

    def __len__(self):
        return len(self.ids_m)

    @classmethod
    def preprocess(cls, pil_img, image_size):
        pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        dt = 1
        idx = self.ids[i]
        idx_m = self.ids_m[i]
        idx_e = self.ids_e[i]
        mask_file = glob(self.masks_dir + idx_m  + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        try:
            next_imgfile = glob(self.imgs_dir + self.ids[i+dt] + '.*')
        except:
            img_file = glob(self.imgs_dir + self.ids[i-dt] + '.*')
            next_imgfile = glob(self.imgs_dir + self.ids[i] + '.*')  
            mask_file = glob(self.masks_dir + self.ids_m[i-dt]  + '.*')
        eventframefile = glob(self.events_dir + idx_e + '.*')
        try:
            next_eventframefile = glob(self.events_dir + self.ids_e[i+dt] + '.*')
        except:
            eventframefile = glob(self.events_dir + self.ids_e[i-dt] + '.*')
            next_eventframefile = glob(self.events_dir + self.ids_e[i] + '.*')
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx_m}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        next_img = Image.open(next_imgfile[0])
        eventframe = Image.open(eventframefile[0])
        next_eventframe = Image.open(next_eventframefile[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        
        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            random.seed(seed)
            torch.manual_seed(seed)
            next_img = self.transform(next_img)
            random.seed(seed)
            torch.manual_seed(seed)
            eventframe = self.transform(eventframe)
            random.seed(seed)
            torch.manual_seed(seed)
            next_eventframe = self.transform(next_eventframe)
            mask = torch.where(mask > 0, 1, 0)
        else:
            img = self.preprocess(img, self.image_size)
            mask = self.preprocess(mask, self.image_size)
            mask = np.where(mask == 1, 1, 0)
            next_img = self.preprocess(next_img, self.image_size)
            eventframe = self.preprocess(eventframe, self.image_size)
            next_eventframe = self.preprocess(next_eventframe, self.image_size)

        return {
            'image': img,
            'eventframe':eventframe,
            'mask': mask,
            'next_image':next_img,
            'next_eventframe':next_eventframe
        }

class evimoDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,events_dir, image_size,transform = None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.events_dir = events_dir
        self.image_size = image_size

        self.ids = [splitext(file)[0] for file in sorted(listdir(imgs_dir))
                    if not file.startswith('.')]
        self.ids_m = [splitext(file)[0] for file in sorted(listdir(masks_dir))
                    if not file.startswith('.')]
        self.ids_e = [splitext(file)[0] for file in sorted(listdir(events_dir))
                    if not file.startswith('.')]
        self.ids_len = len(self.ids)
        self.ids_m_len = len(self.ids_m)
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.transform = transform

    def __len__(self):
        return len(self.ids_m)

    @classmethod
    def preprocess(cls, pil_img, image_size):
        pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):

        idx = self.ids[i]
        # idx_m = int(i/self.ids_len*self.ids_m_len)
        idx_m = self.ids_m[i]
        idx_e = self.ids_e[i]
        
        mask_file = glob(self.masks_dir + idx_m  + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        eventframefile = glob(self.events_dir + idx_e + '.*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx_m}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        eventframe = Image.open(eventframefile[0])
        

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        
        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            
            
            random.seed(seed)
            torch.manual_seed(seed)
            eventframe = self.transform(eventframe)
            
            
            mask = torch.where(mask > 0, 1, 0)
        else:
            img = self.preprocess(img, self.image_size)
            mask = self.preprocess(mask, self.image_size)
            mask = np.where(mask == 1, 1, 0)
            
            eventframe = self.preprocess(eventframe, self.image_size)
            

        return {
            'image': img,
            'eventframe':eventframe,
            'mask': mask
        }