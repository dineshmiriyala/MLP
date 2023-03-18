import os
import pickle
import sys

from model import Neural_net
print("---------------MLP model ----------------")
print("-------Enter only numbers or float--------")

if not os.path.exists('data/trainingData.pkl'):
    exit("Training data not found!")
with open('data/trainingData.pkl' , 'rb') as file:
    data = pickle.load(file)
Xtrain = data['X_train']
ytrain = data['y_train']
Xval = data['X_val']
yval = data['y_val']
Xtest = data['Xtest']
ytest = data['ytest']

model = Neural_net()

user_input = int(input("\n\n\nSelect a option: \n 1: Train \n 2. Loss graph\n 3: Generate Text\n 4. END\n"))
while True:
    if user_input == 1:
        iterations = int(input("Enter number of iterations: \n"))
        model.train(iterations, Xtrain, ytrain)
        user_input = int(input("\n\n\nSelect a option: \n 1: Train \n 2. Loss graph\n 3: Generate Text\n 4. END\n"))
    elif user_input == 2:
        print(f"Current loss is: {model.loss_value}")
        model.graph(model.loss_counter)
        user_input = int(input("\n\n\nSelect a option: \n 1: Train \n 2. Loss graph\n 3: Generate Text\n 4. END\n"))
    elif user_input == 3:
        lines = int(input("Enter number of lines: \n"))
        print()
        model.generate(lines)
        user_input = int(input("\n\n\nSelect a option: \n 1: Train \n 2. Loss graph\n 3: Generate Text\n 4. END\n"))
    else:
        sys.exit()