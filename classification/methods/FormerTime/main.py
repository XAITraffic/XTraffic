import json
import torch
import warnings

warnings.filterwarnings('ignore')
from dataset import Dataset
from FormerTime import FormerTime
from process import Trainer
from args import args
import torch.utils.data as Data


def main():
    torch.set_num_threads(6)
    train_dataset = Dataset(device=args.device, mode='train')
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    test_dataset = Dataset(device=args.device, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')

    model = FormerTime(args)

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    trainer.train()


if __name__ == '__main__':
    main()
