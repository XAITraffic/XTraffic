import torch
import torch.utils.data as Data
from args import Train_data, Test_data


class Dataset(Data.Dataset):
    def __init__(self, device, mode):
        self.device = device
        if mode == 'train':
            self.datas, self.label = Train_data
        else:
            self.datas, self.label = Test_data
        self.mode = mode

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = torch.tensor(self.datas[item]).to(self.device)
        label = self.label[item]
        return data, torch.tensor(label).to(self.device)

    def shape(self):
        return self.datas[0].shape
