# all common library imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# === activation functions === #
alpha = 0.01

activations = {
    'linear'        : (lambda x: x,
                       lambda x: np.ones_like(x)),
    'relu'          : (lambda x: np.maximum(0, x),
                       lambda x: (x>0).astype(float)),
    'leaky_relu'    : (lambda x: np.where(x>0, x, alpha*x),
                       lambda x: np.where(x>0, 1, alpha)),
    'sigmoid'       : (lambda x: 1/(1+np.exp(-x)),
                       lambda x: (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))),
    'tanh'          : (np.tanh,
                       lambda x: 1 - np.tanh(x)**2)
}

# === loss function: mse === #
losses = {
    'mse': (lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
            lambda y_true, y_pred: 2*(y_pred - y_true)/y_true.size)
}