import pickle

import torch
from torch.utils import data
from torch.utils.data import TensorDataset

DATA_PATH = 'data/TIMESERIES_DATA_PARSED/'


def make_timeseries_dataset(opt, mode):
    data_dict = pickle.load(open(DATA_PATH + opt.dataset, 'rb'))
    if mode == 'train':
        dataset = TensorDataset(
            torch.from_numpy(data_dict['X_train']).unsqueeze(1).float(),
            torch.from_numpy(data_dict['y_train']))

    elif mode == 'val':
        dataset = data.TensorDataset(
            torch.from_numpy(data_dict['X_val']).unsqueeze(1).float(),
            torch.from_numpy(data_dict['y_val']))

    elif mode == 'test':
        dataset = data.TensorDataset(
            torch.from_numpy(data_dict['X_test']).unsqueeze(1).float(),
            torch.from_numpy(data_dict['y_test']))

    return dataset
