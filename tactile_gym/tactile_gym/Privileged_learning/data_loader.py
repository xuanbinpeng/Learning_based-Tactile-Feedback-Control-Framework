import torch
import numpy as np
from torch.utils import data

class dataset(data.Dataset):
    def __init__(self, data_set):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        train_data = self.data_set[index]

        data_batch = train_data["train_data"]
        action = train_data["label"]

        return data_batch, action

