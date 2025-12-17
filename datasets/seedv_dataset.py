import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            is_chanle_shafle=False,
            new_order=[]
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.is_chanle_shafle = is_chanle_shafle
        self.new_order = new_order
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        if self.is_chanle_shafle:
            data = data[self.new_order]
        label = pair['label']
        # print(key)
        # print(data)
        # print(label)
        return data / 100, label, key

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
