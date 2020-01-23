
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import torch
from functools import partial
import torch
import os
from PIL import Image
import pandas as pd

attribute_map = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips',
                 'Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows','Chubby',
                 'Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male',
                 'Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin',
                 'Pointy_Nose','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
                 'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace',
                 'Wearing_Necktie','Young']

class CelebA(torch.utils.data.Dataset):
    def __init__(self, opt, data_div):

        if data_div == 0:
            split = 'train'
        elif data_div == 1:
            split = 'valid'
        elif data_div == 2:
            split = 'test'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        if data_div == 0:
            self.transform = transforms.Compose([transforms.Resize((64,73)), transforms.RandomCrop((64,64)), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((64,73)), transforms.CenterCrop((64,64)), transforms.ToTensor(), normalize])

        self.root = opt.dataroot
        self.base_folder = 'celeba'

        self.fn = partial(os.path.join, self.root, self.base_folder)
        csv_file = pd.read_csv(self.fn("list_attr_celeba.txt"))
        splits = pd.read_csv(self.fn("list_eval_partition.txt"))

        target = attribute_map[opt.target_attr]
        print("target attribute ==> ", target)
        target = csv_file[target].values
        target[target == -1] = 0

        filename = csv_file['image_id'].values

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split.lower()]

        mask = slice(None) if split is None else (splits['partition'] == split)

        self.filename = filename[mask]
        self.target = target[mask]

        self.transform = transforms.Compose([transforms.Resize((64,73)), transforms.RandomCrop((64,64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        image = Image.open(self.fn("img_align_celeba",  self.filename[idx]))
        target = self.target[idx]

        image = self.transform(image)

        return image, target.astype('long')
