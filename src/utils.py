#organize text into corpus
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import utils
import numpy as np
import re
import random
import sys
import io
def get_text():
    filename = '../data/heart.txt'
    data = open(filename, 'r', encoding='utf-8').read()
    data = data.lower()
    #output = re.sub(r'\d+', '', raw_text)
    return data

def prepare():
    data = get_text()
    length = 50 #50 chars in a seq
    step = 2
    sentences = []
    next_char = [] #to store the upcoming characters
    for i in range(0, len(data) - length, step):
        sentences.append(data[i : i + length]) #chunks of sentences
        next_char.append(data[i + length]) #next character to predict
    print('your text has %d characters, i split it into %d sentences %d characters each' % (len(data), len(sentences), length))
    return sentences, next_char, length
