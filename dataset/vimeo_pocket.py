import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import logging

class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.training = is_training
        self.data_root = data_root

        self.data_list = []
        self.data_file_list = list(filter(lambda f:f.endswith(".pak"), os.listdir(data_root)))
        for file_name in self.data_file_list:
            f = open(os.path.join(data_root, file_name), 'rb')
            while 1:
                try:
                    self.data_list.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break
        logging.info("init data success")
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        ])
        pass

    def __getitem__(self, index):

        data = self.data_list[index]._sample_1
        # Load images
        img1 = data[0]
        img2 = data[1]
        img3 = data[2]

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            img1 = self.transforms(img1)
            random.seed(seed)
            img2 = self.transforms(img2)
            random.seed(seed)
            img3 = self.transforms(img3)
            # Random Temporal Flip
            # 随机翻转
            if random.random() >= 0.5:
                img1, img3 = img3, img1
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)
        return (img1, img3), img2

    def __len__(self):
        return len(self.data_list)


def get_loader(mode, data_root, batch_size, shuffle):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
