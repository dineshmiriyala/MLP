import pickle
import matplotlib.pyplot as plt
import os


if not os.path.exists('data/stats.pkl'):
    print("Stats not exists")
else:
    with open('data/stats.pkl', 'rb') as file:
        data = pickle.load(file)
    plt.plot(data['loss_values'])
    plt.show()
