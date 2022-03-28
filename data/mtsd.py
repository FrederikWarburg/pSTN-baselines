from builtins import breakpoint
import os
from functools import partial

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class MTSD(torch.utils.data.Dataset):
    def __init__(self, opt, mode):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.bbox_size = opt.bbox_size

        if mode in ['train']:
            transform = [transforms.Resize((224, 224))]
            if opt.data_augmentation:
                transform.append(transforms.RandomHorizontalFlip(0.5))

            transform.extend([transforms.ToTensor(), normalize])

            self.transform = transforms.Compose(transform)
            
        elif mode in ['test','val']:
            self.transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        self.transform_high_res = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), normalize])

        self.root = opt.dataroot
        self.base_folder = 'trafic'

        self.fn = partial(os.path.join, self.root, self.base_folder)

        if mode == "train":
            self.data = pd.read_csv(self.fn("train_data.csv"))
        else:
            self.data = pd.read_csv(self.fn("test_data.csv"))
        self.data = self.data.values

        # make sure we only look at the first max_dataset_size images.
        if mode == 'train':
            size = min(opt.max_dataset_size, len(self.data))
            self.data = self.data[:size]

        self.mapper = {
            'information--pedestrians-crossing--g1': 0,
            'regulatory--no-entry--g1': 1,
            'complementary--chevron-left--g1': 2,
            'complementary--chevron-right--g1': 3,
            'regulatory--yield--g1': 4,
            'regulatory--stop--g1': 5,
            'regulatory--keep-right--g1': 6,
            'information--parking--g1': 7,
            'warning--pedestrians-crossing--g4': 8,
            'regulatory--no-overtaking--g5': 9
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image = Image.open(self.fn("images", self.data[idx, 1].replace(".json", ".jpg")))
        image = np.array(image)

        xmin, xmax, ymin, ymax = self.data[idx, 3:]
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        #y_range = self.bbox_size * (ymax - ymin)
        #x_range = self.bbox_size * (xmax - xmin)

        y_range = int(self.bbox_size * 150)
        x_range = int(self.bbox_size * 150)

        image = image[max(0, ymin - y_range):(ymax + y_range), 
                      max(0, xmin - x_range):(xmax + x_range), :]
        
        image = Image.fromarray(image)

        target = self.mapper[self.data[idx, 2]]

        image_high_res = self.transform_high_res(image)
        image = self.transform(image)

        return image, image_high_res, target
