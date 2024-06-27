import numpy as np
import pandas as pd
from MMDAG import multiDAG_functional, net_fpca
from sklearn.preprocessing import scale
import pickle

if __name__ == '__main__':
    h = {}
    T = 24
    rho = 1
    K = 2
    lambda1 = 0.001
    h[0] = np.load('district_3_DAGdata.npy')
    h[0] = h[0].transpose(0, 2, 1)
    h[1] = np.load('district_4_DAGdata.npy')
    h[1] = h[1].transpose(0, 2, 1)
    h[2] = np.load('district_7_DAGdata.npy')
    h[2] = h[2].transpose(0, 2, 1)
    h[3] = np.load('district_8_DAGdata.npy')
    h[3] = h[3].transpose(0, 2, 1)
    P_id = {}
    P_all = 18
    for i in range(4):
        P_id[i] = range(P_all)
        scale_matrix = np.zeros_like(h[i], dtype=np.float64)
        for j in range(P_all):
            scale_matrix[:, j, :] = h[i][:, j, :] / np.std(h[i][:, j, :])
        h[i] = scale_matrix
    a, v = net_fpca(h, K=K)
    E_est, G_est = multiDAG_functional(a.copy(), lambda1=lambda1, rho=rho, P_id=P_id, P_all=P_all)
    file_name = f'./district_DAG_lambda_{lambda1}_rho_{rho}_K_{K}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump((E_est, G_est), file)
