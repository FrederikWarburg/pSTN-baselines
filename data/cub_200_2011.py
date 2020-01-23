import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.utils import make_affine_parameters
from torch import distributions
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def scale_keep_ar_min_fixed(img, fixed_min):
    ow, oh = img.size

    if ow < oh:

        nw = fixed_min

        nh = nw * oh // ow

    else:

        nh = fixed_min

        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)


def transform_image_manual(x, opt):
    # we always applies the same data augmentation during training
    num_param = opt.num_param #2 if opt.fix_scale_and_rot else 4

    gaussian = distributions.normal.Normal(0, 1)  # split up the multivariate Gaussian into 1d Gaussians
    epsilon = gaussian.sample(sample_shape=torch.Size([num_param]))

    random_params = epsilon * opt.sigma
    if num_param == 4: random_params[1] += 1 # scale is centered around 1
    if num_param == 6:
        random_params[0] += 1 # scale is centered around 1
        random_params[4] += 1 # scale is centered around 1
    theta = make_affine_parameters(random_params.unsqueeze(0))

    x = x.unsqueeze(0)
    grid = F.affine_grid(theta, x.size())  # makes the flow field on a grid
    x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
    return x_transformed.squeeze(0)

def transform(opt):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    transform_list = []
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, opt.smallest_size)))
    if opt.is_train:
        if opt.data_augmentation:
            if opt.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.7))
                transform_list.append(transforms.RandomCrop((opt.crop_size, opt.crop_size)))
            else:
                transform_list.append(transforms.CenterCrop((opt.crop_size, opt.crop_size)))
            transform_list.append(transforms.ToTensor())
            #transform_list.append(lambda img: transform_image_manual(img, opt))
        else:
            transform_list.append(transforms.CenterCrop((opt.crop_size, opt.crop_size)))
            transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.CenterCrop((opt.crop_size, opt.crop_size)))
        #transform_list.append(transforms.Resize((448, 448)))
        transform_list.append(transforms.ToTensor())

    transform_list.append(normalize)

    return transforms.Compose(transform_list)

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, opt, data_div):

        self.root = os.path.expanduser(opt.dataroot)
        self.transform = transform(opt)
        self.loader = default_loader
        self.train = data_div > 0
        self.val = data_div == 1
        self.num_classes = opt.num_classes
        self.trainval_split = opt.trainval_split

        if opt.download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        if not os.path.isfile(os.path.join(self.root, 'CUB_200_2011', 'train_val_split.txt')): self._make_validation_set()
        train_val_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_val_split.txt'), sep=' ', names=['img_id', 'is_validation_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.data = self.data.merge(train_val_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]

            if self.trainval_split:
                self.data = self.data[self.data.is_validation_img == 1] if self.val else self.data[self.data.is_validation_img == 0]
            else:
                if self.val: self.data = self.data[self.data.is_validation_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.data = self.data[self.data.target < self.num_classes + 1]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)


    def _make_validation_set(self):

        train_val_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])

        train_val_split["is_training_img"][train_val_split["is_training_img"] == 1] = np.random.binomial(1, 0.1, sum(train_val_split["is_training_img"] == 1))

        np.savetxt(os.path.join(self.root, 'CUB_200_2011', 'train_val_split.txt'), train_val_split.values, fmt='%u %u')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
