# this is an extension of preprocessing script in Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)
#imports
import warnings
import regex
import torch
import os
import random
import pickle
random.seed(9968)

warnings.filterwarnings('ignore')

lines = open('reddit_convos.txt' , 'r').read().splitlines()

def text_clean(lines):
  new_string = []
  pattern = regex.compile(r'\b\w*(\w)\1{2,}\w\b')
  for line in lines:
    line = regex.sub(pattern , '' , line)
    new_string.append(regex.sub(r"(.)\1+", r"\1", line))
  return new_string

lines = text_clean(lines)
words = []
for line in lines:
    words.extend(line.split(' '))
words = sorted(list(set(words)))

#encoder and decoder
words_to_integers = {word: integer + 1 for integer , word in enumerate(words)}
words_to_integers['nigga'] = len(words_to_integers)
words_to_integers['dinesh'] = len(words_to_integers)
words_to_integers['.'] = 0
integer_to_words = {word: integer for integer, word in words_to_integers.items()}

"""In this project I am taking three words to predict the next one. That means the block size would be three. I am 
keeping it constant and not giving any flexibility to change as it is one of the core components of model and 
everything should be changed if we change this one parameter."""
def build_datset(data):
    block_size = 3  # context length
    X,y = [] , []
    for lines in data:
        context = [0] * block_size
        for word in lines.split() + list('.'):
            index = words_to_integers[word]
            X.append(context)
            y.append(index)
            context = context[1:] + [index]
    X = torch.tensor(X)
    y = torch.tensor(y)
    return X,y

"""I am creating datasets with 80% of data for training and 10% each for both validation and testing.
ALl these will be saved in pickle files, for easier access."""
random.shuffle(lines)
n1 = int(0.8*len(lines))
n2 = int(0.9*len(lines))


#saving datasets
if not os.path.exists('trainingData.pkl'):
    open('trainingData.pkl' , 'w')
    Xtrain, ytrain = build_datset(lines[:n1])
    Xval, yval = build_datset(lines[n1:n2])
    Xtest, ytest = build_datset(lines[n2:])
    data = {}
    data['X_train'] = Xtrain
    data['y_train'] = ytrain
    data['X_val'] = Xval
    data['y_val'] = yval
    data['Xtest'] = Xtest
    data['ytest'] = ytest
    with open('trainingData.pkl' , 'wb') as file:
        pickle.dump(data , file)
        print('Dataset files created successfully')

#saving encoders and decoders
if not os.path.exists('encode_decode.pkl'):
    open('encode_decode.pkl' , 'w')
    encode_decode = {}
    encode_decode['encode'] = words_to_integers
    encode_decode['decode'] = integer_to_words
    with open('encode_decode.pkl' , 'wb') as file:
        pickle.dump(encode_decode , file)
        print('Encode and decode engine created successfully')