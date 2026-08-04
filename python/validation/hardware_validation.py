import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

gain = 120000

l2_neg_raw = [8.01,8.01,9.5,7.32,7.32,7.89]
l2_pos_raw = [7.82,8.3,7.64,9.25,8.57,11.78]

sig_in=np.array(l2_pos_raw) - np.array(l2_neg_raw)
print(sig_in)
sig_in = sig_in / 1000000
sig_in = sig_in * gain
result = sigmoid(sig_in)
print(result)

