import os
import pickle
from model import Neural_net
import numpy

if not os.path.exists('data/trainingData.pkl'):
    exit("Training data not found!")
with open('data/trainingData.pkl' , 'rb') as file:
    data = pickle.load(file)
    print("#####Training data loaded#####")
Xtrain = data['X_train']
ytrain = data['y_train']
Xval = data['X_val']
yval = data['y_val']

model = Neural_net()

model.train(30000, Xtrain, ytrain)

model.generate(10)