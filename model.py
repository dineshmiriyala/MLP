#this is build on the dataset from Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import regex
import re
import random
random.seed(9968)

class dataset():
    def __init__(self , lines):
        self.lines = lines
    def
