import numpy as np
from scipy.io import arff
import pickle as pkl 

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='../../', folder='Cricket'):
    datas = pkl.load(open(Path + 'data/traffic_data.pkl', 'rb'))
    TRAIN_LABEL = datas['y'][0] 
    TRAIN_DATA = datas['x'][0].reshape(-1, 3, 24)[:, :].transpose(0, 2, 1)
    TEST_LABEL = datas['y'][1] 
    TEST_DATA = datas['x'][1].reshape(-1, 3, 24)[:, :].transpose(0, 2, 1)
    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TEST_DATA), np.array(TEST_LABEL)]

