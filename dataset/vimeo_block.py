import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2
class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.training = is_training
        self.data_root = data_root

        # 按照sep—list格式读取
        self.datalist = []
        def data_path_get(file_name):
            data_path = os.path.join(data_root, file_name)
            list(map(lambda f: self.datalist.append(file_name+"/"+f),os.listdir(data_path)))
        list(map(data_path_get, filter(lambda f:os.path.isdir(data_root + "/" + f), os.listdir(data_root))))

        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
        pass

    def __getitem__(self, index):
        imgpath = os.path.join(self.data_root, self.datalist[index])

        file_names = list(os.listdir(imgpath))
        min_index = min(list(map(lambda n:int(n.split(".")[0][-1]), file_names)))
        data_len = len(file_names)
        label_index = random.choice(list(range(data_len)[1+min_index:data_len+min_index-1]))

        imgpaths = [imgpath + '/im{}.png'.format(label_index-1), imgpath + '/im{}.png'.format(label_index), imgpath + '/im{}.png'.format(label_index+1)]

        # Load images
        # img1 = cv2.imread(imgpaths[0])
        # img2 = cv2.imread(imgpaths[1])
        # img3 = cv2.imread(imgpaths[2])
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

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
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)
        return (img1, img3), img2

    def __len__(self):
        return len(self.datalist)


def get_loader(mode, data_root, batch_size, shuffle):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
