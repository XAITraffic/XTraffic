import torch
import torch.nn as nn
import numpy as np
data = []
class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs.float())  # B * N
        data.append(outputs.detach().cpu().numpy())
        np.save('outputs.npy', np.concatenate(data))
        labels = labels.view(-1).long()
        loss = self.ce(outputs, labels)
        return loss


class BCE:
    def __init__(self, model):
        self.model = model
        self.bce = nn.BCELoss(reduction='none')

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs)  # B * N
        weight = torch.ones(outputs.shape[0]).float().to(outputs.device)
        loss = self.bce(outputs.view(-1), labels.float())
        loss = torch.mean(weight * loss)
        return loss
