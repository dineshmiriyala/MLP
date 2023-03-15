# this is build on the dataset from Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)

import torch
import pickle
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
import random

random.seed(9968)

warnings.filterwarnings('ignore')


# data imports
def import_dataset():
    if os.path.exists('data/encode_decode.pkl'):
        with open('data/encode_decode.pkl', 'rb') as file:
            try:
                encode_decode = pickle.load(file)
            except Exception:
                print(
                    "Something went wrong while loading encode_decode engine. "
                    "Try running data/preprocess.py file. \n")
        return encode_decode


encode, decode = (import_dataset())['encode'], (import_dataset())['decode']

class Neural_net():
    def __init__(self):
        self.prob = None
        self.gen = torch.Generator(device=device).manual_seed(9968)
        self.W1 = None
        self.W2 = None
        self.B1 = None
        self.B2 = None
        self.loss_value = None
        self.params()
        self.loss_counter, self.lr_counter = self.load_stats()
        self.lr = 0.1

    def params(self):
        if not os.path.exists('data/params.pkl'):
            open('data/params.pkl', 'w')
        with open('data/params.pkl', 'rb') as file:
            try:
                params = pickle.load(file)
                self.W1 = params['W1']
                self.W2 = params['W2']
                self.B1 = params['B1']
                self.B2 = params['B2']
                self.prob = params['prob']
                print("Model loaded successfully! \n")
                print(f"Current loss: {params['loss_value']}")
            except EOFError:
                print("Pre-trained model not found. Try training the model first. \n")
                self.prob = torch.randn((len(encode), 60), generator=self.gen,
                                        device=device, requires_grad=True)
                self.W1 = torch.randn((180, 200), generator=self.gen,
                                      device=device, requires_grad=True)
                self.B1 = torch.randn(200, generator=self.gen,
                                      device=device, requires_grad=True)
                self.W2 = torch.randn((200, len(encode)), generator=self.gen,
                                      device=device, requires_grad=True)
                self.B2 = torch.randn(len(encode), generator=self.gen,
                                      device=device, requires_grad=True)
                print("Parameters Created.")

    def encode(self, word):

        return encode[word]

    def decode(self, integer):
        return decode[integer]
    def clear_params(self):
        """Clearing all the gradients."""
        self.prob.grad = None
        self.B1.grad = None
        self.B2.grad = None
        self.W1.grad = None
        self.W2.grad = None

    def backward(self):
        self.clear_params()
        self.loss_value.backward()

    def forward(self, values):
        return self.prob[values]

    def hidden(self, embedded):
        hidden = torch.tanh(embedded.view(-1, 180) @ self.W1 + self.B1)
        logits = hidden @ self.W2 + self.B2
        return logits

    def update(self, iteration):
        self.lr = 0.1 if iteration < 100000 else 0.01
        self.prob.data += -self.lr * self.prob.grad
        self.B1.data += -self.lr * self.B1.grad
        self.B2.data += -self.lr * self.B2.grad
        self.W1.data += -self.lr * self.W1.grad
        self.W2.data += -self.lr * self.W2.grad

    def train(self, iterations, X_train, y_train):
        bar = iterations // 10
        progress = 0
        print("Progress: ", end = "")
        for iteration in range(iterations):
            #progress bar
            if iteration % bar == 0:
                print(f'{progress}%__', end = '')
                progress += 10
            # mini batches construction
            index = torch.randint(0, X_train.shape[0], (128,))
            # forward pass
            embedded = self.forward(X_train[index])
            # hidden layer
            logits = self.hidden(embedded)
            self.loss_value = F.cross_entropy(logits, y_train[index])
            # backward pass
            self.backward()
            # update params
            self.update(iteration)
            self.loss_counter.append(self.loss_value.log10().item())
            self.lr_counter.append(self.lr)
        self.save_model()
        self.save_stats()
        print(f"Current loss------> {self.loss_value}")

    def generate(self, number):
        if self.loss_value < 10:
            for _ in range(number):
                out = []
                context = [0] * 3
                while True:
                    embedded = self.forward(torch.tensor([context]))
                    logits = self.hidden(embedded)
                    probability = F.softmax(logits, dim=1)
                    index = torch.multinomial(probability, num_samples=1).item()
                    context = context[1:] + [index]
                    out.append(index)
                    if index == 0:
                        break
                print(' '.join(self.decode(integer) for integer in out))
        else:
            print("Model is not good enough to sample from.")
            print(f"Current loss: {self.loss_value}\n")

    def testing(self, X, y):
        loss = []
        for i in range(X.shape[0] // 256):
            # mini batch construction
            index = torch.randint(0, X.shape[0], (256,))
            embedded = self.forward(X[index])
            logits = self.hidden(embedded)
            loss.append(F.cross_entropy(logits, y[index]))
        print(f"The loss for given dataset is: {sum(loss) / len(loss)}")

    def save_model(self):
        if not os.path.exists('data/params.pkl'):
            open('data/params.pkl' , 'w')
        with open('data/params.pkl' , 'wb') as file:
            params = {}
            params['W1'] = self.W1
            params['W2'] = self.W2
            params['B1'] = self.B1
            params['B2'] = self.B2
            params['prob'] = self.prob
            params['loss_value'] = self.loss_value
            pickle.dump(params, file)
            print("Model updated and saved successfully! \n")

    def save_stats(self):
        if not os.path.exists('data/stats.pkl'):
            open('data/stats.pkl', 'w')
        data = {}
        data['loss_values'] = self.loss_counter
        data['lr'] = self.lr_counter
        with open('data/stats.pkl', 'wb') as file:
            pickle.dump(data, file)
    def load_stats(self):
        if not os.path.exists('data/stats.pkl'):
            open('data/stats.pkl', 'w')
            return [], []
        with open('data/stats.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['loss_values'], data['lr']