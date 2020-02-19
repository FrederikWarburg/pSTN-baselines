import os
import pickle
from glob import glob

import numpy as np

DATA_PATH = '../data_augmentation/time_series/DATA_PARSED/'


# make lists of data sets names
def make_ds_list(PATH):
    ds_list = []
    for folder_PATH in glob(PATH + '*/'):
        ds_list.append(folder_PATH.split("/")[-2])
    ds_list = np.sort(ds_list)
    return ds_list


def get_nr_classes_and_features(ds):
    data_dict = pickle.load(open(DATA_PATH + ds, 'rb'))
    num_classes = max(max(np.unique(data_dict['y_train'])), max(np.unique(data_dict['y_val']))) + 1
    num_features = data_dict['X_train'].shape[1]
    return num_classes, num_features


def make_onehot_labels(y, num_classes):
    y = y.flatten()
    length = len(y)
    y_onehot = np.zeros((length, num_classes))
    y_onehot[np.arange(length), y] = 1
    return y_onehot


def make_label_lookup(y_train):
    # make a dictionary of lists that, for each class,
    # contains indices of datapoints from that class in the training set
    classes = np.unique(y_train)
    class_indices = {}
    for label in classes:
        class_indices[label] = np.where(y_train == label)[0]
    return class_indices


def fix_class_indexing(y_train, y_test):
    # Make sure the labels are integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    idx = 0
    for label in np.unique(y_train):
        y_train[np.where(y_train == label)] = idx
        y_test[np.where(y_test == label)] = idx
        idx += 1
    return y_train, y_test


def load_thetas_per_class(PATH, ds):
    all_trafos = {}
    for file in os.listdir(PATH + ds):
        if file.startswith('transformations'):
            suffix = file.split('transformations')[1]
            label = int(suffix.split('.')[0])
            label_path = PATH + ds + '/' + file
            trafos_here = pickle.load(open(label_path, 'rb'))
            print('Loaded ', trafos_here.shape[0], 'valid trafos.')
            # remove nan's
            all_trafos[label] = trafos_here[~np.isnan(trafos_here).any(axis=1)]
            print('Found ', trafos_here.shape[0], 'valid trafos for label', label)
    return all_trafos
