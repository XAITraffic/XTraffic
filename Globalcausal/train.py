import numpy as np


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = params
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, grads):
        if self.m is None:
            self.m = np.zeros_like(self.params)
            self.v = np.zeros_like(self.params)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads ** 2 - self.v)
        self.params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
