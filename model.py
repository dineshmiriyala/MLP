#this is build on the dataset from Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)

import torch
import pickle
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
random.seed(9968)

warnings.filterwarnings('ignore')
#data imports
def import_dataset():
    if os.path.exists('data/encode_decode.pkl') and os.path.exists('data/trainingData.pkl'):
        with open('data/encode_decode.pkl' , 'rb') as file:
            try:
                encode_decode = pickle.load(file)
            except Exception:
                print("Something went wrong while loading encode_decode engine. Try running data/preprocess.py file. \n")
        with open('data/trainingData.pkl' , 'rb') as file:
            try:
                data = pickle.load(file)
            except Exception:
                print("Something went wrong while loading datasets. Try running data/preprocess.py file. \n")
        print("data file loaded successfully! \n")
        return encode_decode , data

encode_decode , data = import_dataset()


class dataset():
    def __init__(self , lines):
        self.data = data
        self.gen = torch.Generator().manual_seed(9968)
        

    def encode(self , word):
        return encode_decode.encode[word]

    def decode(self , integer):
        return encode_decode.decode[integer]
    def



