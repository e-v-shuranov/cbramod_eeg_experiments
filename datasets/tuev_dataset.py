import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            files,
            n_chanels,
            is_chanle_shafle = False,
            new_order = []
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.files = files
        self.n_chanels = n_chanels
        self.is_chanle_shafle = is_chanle_shafle
        self.new_order = new_order
        # self.new_order = [ 3,  8,  7,  6,  9,  1, 14,  5, 13, 12,  0,  2, 10,  4, 15, 11]
        # self.new_order = [13, 15, 1, 8, 10, 0, 2, 3, 11, 6, 9, 14, 7, 4, 5, 12]
        # self.new_order = [ 3, 11, 10,  6,  1,  4,  7, 15, 12,  0, 14,  8,  2,  9, 13,  5]
        # self.new_order = [ 9,  6, 13, 15,  1,  7, 12,  0,  2, 14,  5, 10, 11,  4,  3,  8]
        # self.new_order = [10, 14,  6,  2, 15,  9,  0,  5, 13, 11, 12,  8,  7,  1,  4,  3]
        # self.new_order = [13,  8, 14,  4,  9,  0,  2, 11, 12,  6,  5, 10,  7,  3, 15,  1]
        # self.new_order = [ 5, 12,  6,  4,  1,  2, 11, 15, 10,  3,  0, 14,  7,  9, 13,  8]
        # self.new_order = [10, 13,  6,  4, 11, 15,  1, 14,  5,  3,  7,  0,  2,  9,  8, 12]
        # self.new_order = [ 8, 13, 12, 10, 11,  1,  0,  5, 14,  7,  4, 15,  3,  6,  2,  9]
        # self.new_order = [ 4,  5, 11,  7,  3, 10,  8, 13,  9,  0, 15, 14,  2,  1,  6, 12]


    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
        data = data_dict['signal']
        label = int(data_dict['label'][0]-1)
        # data = signal.resample(data, 1000, axis=-1)
        data = data[:self.n_chanels].reshape(self.n_chanels, 5, 200)
        if self.is_chanle_shafle:
            data = data[self.new_order]
        # data = data.reshape(self.n_chanels, 5, 200)
        return data/100, label, file

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        xs, ys, files = zip(*batch)
        return to_tensor(x_data), to_tensor(y_label).long(),  list(files)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_files = os.listdir(os.path.join(self.datasets_dir, "processed_train"))
        val_files = os.listdir(os.path.join(self.datasets_dir, "processed_eval"))
        test_files = os.listdir(os.path.join(self.datasets_dir, "processed_test"))

        train_set = CustomDataset(os.path.join(self.datasets_dir, "processed_train"), train_files, self.params.n_chanels, is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "processed_eval"), val_files, self.params.n_chanels, is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "processed_test"), test_files, self.params.n_chanels, is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
