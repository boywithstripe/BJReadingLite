from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
from .randaugment import RandomAugment
import random

class data_randAugment(data.Dataset):
    def __init__(self, config, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        char_file = config.DATASET.CHAR_FILE
        with open(char_file, 'rb') as file:
            char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                if(indices[-1]=='\n'):
                    del indices[-1]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        if img_name.split('.')[-1] == "txt":
            img_name = img_name.split('.')[0]+'.jpg'
        img = cv2.imread(os.path.join(self.root, img_name))

        # 随机进行数据增强
        rate = random.random()
        if(rate>0.8):
            a = RandomAugment()   # randaugment 图像增强
            img = a(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx








