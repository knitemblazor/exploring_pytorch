import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

m = nn.Linear(20, 30)
inp = 20*[i for i in range(128)]
inp_numpy = m.weight.detach().numpy()

a = [i for i in inp_numpy[0]]
b = [i for i in range(-10,10)]
plt.plot(b,a)
plt.show()
