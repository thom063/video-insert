import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
        pass
        

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        file_names = list(os.listdir(imgpath))
        min_index = min(list(map(lambda n:int(n.split(".")[0][-1]), file_names)))
        data_len = len(file_names)
        label_index = random.choice(list(range(data_len)[1+min_index:data_len+min_index-1]))

        imgpaths = [imgpath + '/im{}.png'.format(label_index-1), imgpath + '/im{}.png'.format(label_index), imgpath + '/im{}.png'.format(label_index+1)]

        # Load images
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
        #
        # imgs = torch.concat([img1.unsqueeze(0), img3.unsqueeze(0)],dim=0)
        
        return (img1, img3), img2

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        return 0


def get_loader(mode, data_root, batch_size, shuffle):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, num_workers=num_workers, pin_memory=True)
