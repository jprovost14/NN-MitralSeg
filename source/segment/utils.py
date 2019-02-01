import numpy as np
from torch.utils.data.dataset import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, dataset):

        self.__dataset = dataset

    def __getitem__(self, index):
        data = self.__dataset[index]

        pixel_id = np.array(data[0], dtype=int)
        frame_id = np.array(data[1], dtype=int)
        target = np.array(data[2])

        return torch.from_numpy(pixel_id), torch.from_numpy(frame_id), torch.from_numpy(target)

    def __len__(self):
        self.__size = self.__dataset.shape[0]
        return self.__size
