import pickle
import matplotlib.pyplot as plt
import os
import math
import numpy
import torch.cuda

if not os.path.exists('data/stats.pkl'):
    print("Stats not exists")
with open('data/stats.pkl', 'rb') as file:
    data = pickle.load(file)
print(len(data['loss_values']))
plt.plot(data['loss_values'])
plt.show()