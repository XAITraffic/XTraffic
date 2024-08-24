import numpy as np
import utils
import igraph
from skfda.representation.basis import Fourier

def generate_multiDAG(N=1000, P=10, K=2, s=20, n_task=5, T=100, seed=23):
    #utils.set_random_seed(seed)
    graph_type = 'ER'
    T_true= utils.simulate_dag(P, s, graph_type)
    E_true = np.zeros((n_task, P, P))
    W_true = np.zeros((n_task, P, P))
    fourier_basis = Fourier((0, 1), n_basis=K, period=1)
    s = fourier_basis(np.arange(0, 1, 1 / T))
    g = np.zeros((n_task, N, P, T))
    h = np.zeros((n_task, N, P, T))
    true_a = np.zeros((n_task, N, P, K))
    for task in range(n_task):
        E_true[task] = utils.simulate_dag(P, int(P * (P - 1) / 2), graph_type) * T_true
        W_true[task] = utils.simulate_parameter(E_true[task])
        G = igraph.Graph.Adjacency(E_true[task])
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == P
        for i in range(N):
            delta_i = np.zeros((P, K))
            for j in ordered_vertices:
                parents = G.neighbors(j, mode=igraph.IN)
                mean = np.zeros(K)
                for k in parents:
                    mean += (delta_i[k, :] * W_true[task, k, j]).reshape(K)
                delta_i[j, :] = np.random.multivariate_normal(mean=mean, cov=np.identity(K))
            for j in range(P):
                true_a[task, i, j, :] = delta_i[j]
                for l in range(K):
                    g[task, i, j, :] += (delta_i[j, l] * s[l, :]).reshape((T,))
                for t in range(T):
                    h[task, i, j, t] = np.random.normal(loc=g[task, i, j, t], scale=0.01)
    return g, h, E_true, W_true, T_true, true_a

