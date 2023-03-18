# this is build on the dataset from Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)

import torch
import pickle
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
import random
from tqdm import tqdm

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
        self.B2 = None
        self.loss_value = None
        self.BatchGain = None
        self.BatchBias = None
        self.Batchmean = None
        self.Batchstd = None
        self.loss_counter, self.lr_counter = self.load_stats()
        self.lr = 0.1
        self.params()

    def params(self):
        if not os.path.exists('data/params.pkl'):
            open('data/params.pkl', 'w')
        with open('data/params.pkl', 'rb') as file:
            try:
                params = pickle.load(file)
                self.W1 = params['W1']
                self.W2 = params['W2']
                self.B2 = params['B2']
                self.prob = params['prob']
                self.loss_value = params['loss_value']
                self.BatchBias = params['BatchBias']
                self.BatchGain = params['BatchGain']
                self.Batchmean = params['BatchMean']
                self.Batchstd = params['BatchStd']
                print("Model loaded successfully! \n")
                print(f"Current loss: {params['loss_value']} \n")
            except EOFError:
                print("Pre-trained model not found. Try training the model first. \n")
                self.prob = torch.randn((len(encode), 60), device=device, requires_grad=True)
                self.W1 = torch.ones((180, 200), device=device, requires_grad=True)
                self.W2 = torch.ones((200, len(encode)), device=device, requires_grad=True)
                self.B2 = torch.zeros(len(encode), device=device, requires_grad=True)
                self.BatchBias = torch.zeros((1, 200), device=device, requires_grad=True)
                self.BatchGain = torch.ones((1, 200), device=device, requires_grad=True)
                self.Batchmean = torch.zeros((1, 200), device=device, requires_grad=True)
                self.Batchstd = torch.ones((1, 200), device=device, requires_grad=True)
                print("Parameters Created.")

    def encode(self, word):
        return encode[word]

    def decode(self, integer):
        if integer == 41355:
            return 'India'
        return decode[integer]

    def clear_params(self):
        """Clearing all the gradients."""
        self.prob.grad = None
        self.B2.grad = None
        self.W1.grad = None
        self.W2.grad = None
        self.BatchGain.grad = None
        self.BatchBias.grad = None

    def update(self, iteration):
        self.lr = 0.1 if iteration < 100 else 0.01
        self.prob.data += -self.lr * self.prob.grad
        self.B2.data += -self.lr * self.B2.grad
        self.W1.data += -self.lr * self.W1.grad
        self.W2.data += -self.lr * self.W2.grad
        self.BatchBias.data += -self.lr * self.BatchBias.grad
        self.BatchGain.data += -self.lr * self.BatchGain.grad

    def train(self, iterations, X_train, y_train):
        for iteration in tqdm(range(iterations)):
            # mini batches construction
            index = torch.randint(0, X_train.shape[0], (128,))
            # forward pass
            emb = self.prob[X_train[index]]
            # linear layer
            embedded = emb.view(emb.shape[0], -1) @ self.W1
            # batch norm layer
            batchnorm_mean = embedded.mean(0, keepdim = True)
            batchnorm_std = embedded.std(0, keepdim=True)
            # print(type(self.BatchGain), type(embedded), type(batchnorm_mean))
            hidden_preact = (self.BatchGain * (embedded - batchnorm_mean)) / (batchnorm_std + self.BatchBias)
            # updating running batch parameters
            with torch.no_grad():
                self.Batchmean = 0.99*self.Batchmean + 0.01 * batchnorm_mean
                self.Batchstd = 0.99*self.Batchstd + 0.01 * batchnorm_mean
            # Non-linear layer
            hidden = torch.tanh(hidden_preact)
            logits = hidden @ self.W2 + self.B2
            self.loss_value = F.cross_entropy(logits, y_train[index])
            # backward pass
            self.clear_params()
            self.loss_value.backward()
            # update params
            self.update(iteration)
            self.loss_counter.append(self.loss_value.log10().item())
            self.lr_counter.append(self.lr)
        self.save_model()
        self.save_stats()
        print(f"Current loss------> {self.loss_value}")
        print("Graph for current loss")
        self.graph(self.loss_counter)

    def generate(self, number):
        if self.loss_value < 10:
            for _ in range(number):
                out = []
                context = [0] * 3
                while True:
                    emb = self.prob[torch.tensor([context])]
                    # linear layer
                    embedded = emb.view(emb.shape[0], -1) @ self.W1
                    # batch norm layer
                    hidden_preact = (self.BatchGain * (embedded - self.Batchmean)) / (self.Batchstd + self.BatchBias)
                    # non-linear layer
                    hidden = torch.tanh(hidden_preact)
                    logits = hidden @ self.W2 + self.B2
                    probability = F.softmax(logits, dim=1)
                    index = torch.multinomial(probability, num_samples=1).item()
                    context = context[1:] + [index]
                    out.append(index)
                    if index == 0:
                        break
                print(' '.join(self.decode(integer) for integer in out))
        else:
            print("Model is not good enough to sample from.")
            print(f"Current loss: {self.loss_value}\n\n")

    def testing(self, X, y):
        with torch.no_grad():
            loss = []
            for i in tqdm(range(X.shape[0] // 256)):
                # mini batch construction
                index = torch.randint(0, X.shape[0], (256,))
                emb = self.prob[X[index]]
                # linear layer
                embedded = emb.view(emb.shape[0], -1) @ self.W1
                # batch norm layer
                hidden_preact = (self.BatchGain * (embedded - self.Batchmean)) / (self.Batchstd + self.BatchBias)
                # non-linear layer
                hidden = torch.tanh(hidden_preact)
                logits = hidden @ self.W2 + self.B2
                loss.append(F.cross_entropy(logits, y[index]))
            print(f"The loss for given dataset is: {sum(loss) / len(loss)}")
            print("Graph for loss values in testing set: ")
            self.graph(loss)

    def graph(self, values):
        plt.plot(torch.tensor(values).view(-1,len(values) // 110).mean(1))
        plt.title("loss values")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
        
    def save_model(self):
        if not os.path.exists('data/params.pkl'):
            open('data/params.pkl', 'w')
        with open('data/params.pkl', 'wb') as file:
            params = {'W1': self.W1, 'W2': self.W2, 'B2': self.B2, 'prob': self.prob
                , 'loss_value': self.loss_value, 'BatchGain': self.BatchGain,
                      'BatchBias' : self.BatchBias, 'BatchMean': self.Batchmean,
                      'BatchStd': self.Batchstd}
            pickle.dump(params, file)
            print("Model updated and saved successfully! \n")

    def save_stats(self):
        if not os.path.exists('data/stats.pkl'):
            open('data/stats.pkl', 'w')
        data = {'loss_values': self.loss_counter, 'lr': self.lr_counter}
        with open('data/stats.pkl', 'wb') as file:
            pickle.dump(data, file)

    def load_stats(self):
        if not os.path.exists('data/stats.pkl'):
            open('data/stats.pkl', 'w')
            return [], []
        with open('data/stats.pkl', 'rb') as file:
            try:
                data = pickle.load(file)
                return data['loss_values'], data['lr']
            except Exception:
                return [], []
