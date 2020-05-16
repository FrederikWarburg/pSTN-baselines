import numpy as np
import os

dataset_list = ['MNIST10', 'MNIST30', 'MNIST100', 'MNIST1000', 'MNIST3000', 'MNIST10000']
folds = range(5)

for dataset in dataset_list:
    print('Processing dataset', dataset)
    all_train_indices = []
    for fold in folds:
        fold_indices = np.load('../data/subset_indices/' + dataset + '_train_indices_fold_%s.npy'% fold)
        all_train_indices.extend(fold_indices)
    all_train_indices = set(all_train_indices)
    try:
        val_indices = set(np.load('../data/subset_indices/%s_validation_indices.npy' %dataset))
    except:
        print('Validation indices missing for', dataset)
    print('All train indices are', len(list(all_train_indices)),
          '\n Validation indices are', len(list(val_indices)),
          '\n overlap is ', len(list(all_train_indices.intersection(val_indices))))
