from Datasets.unimatch_utils import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, dataset, img_size, use_aug=False,  
        data_path='./proceeded_data/train/'):
        super(SemiDataset, self).__init__()
        self.dataset = dataset
        self.root_dir = data_path
        self.size = img_size

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')
        
        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5
        
        img = Image.fromarray(np.uint8(img_data))
        mask = Image.fromarray(np.uint8(label_data))


        img, mask = img.resize((224, 224)), mask.resize((224, 224))
        img, mask = hflip(img, mask, p=0.5)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        return normalize(img_w), img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.dataset)