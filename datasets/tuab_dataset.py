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
            mode='train',
            is_chanle_shafle=False,
            new_order=[]
    ):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]
        self.is_chanle_shafle = is_chanle_shafle
        self.new_order = new_order


    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']
        label = data_dict['y']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(16, 10, 200)
        if self.is_chanle_shafle:
            data = data[self.new_order]
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
        train_set = CustomDataset(self.datasets_dir, mode='train', is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)
        val_set = CustomDataset(self.datasets_dir, mode='val', is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)
        test_set = CustomDataset(self.datasets_dir, mode='test', is_chanle_shafle = self.params.is_chanle_shafle,new_order=self.params.new_order)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
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
