#organize text into corpus
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import re
import random
import sys
import io

def prepare():
    #enclosing this so that we don't have to ask for user input twice
    def get_text():
        files = {'1':'prest_i_nakaz','2':'heart','3':'metamorphosis','4':'iceland'}
        print('choose text:\n')
        print('\t1: russian existentialism\n')
        print('\t2: apocalypse now but in text form\n')
        print('\t3: austrian existentialism\n')
        print('\t4: icelandic short story - note: short and guaranteed to generate gobbledygook\n')
        text = input('type: ')
        name = files[text]
        filename = '../data/'+name+'.txt'
        data = open(filename, 'r', encoding='utf-8').read()
        data = data.lower()
        data = re.sub(r'\d+', '', data)
        data = re.sub(r'^([0-9]+)|([IVXLCM]+)\\.?$', '', data)
        return data, name
    data, name = get_text()
    length = 50 #50 chars in a seq
    step = 2
    sentences = []
    next_char = [] #to store the upcoming characters
    for i in range(0, len(data) - length, step):
        sentences.append(data[i : i + length]) #chunks of sentences
        next_char.append(data[i + length]) #next character to predict
    print('your text has %d characters, i split it into %d sentences %d characters each' % (len(data), len(sentences), length))
    return name, data, sentences, next_char, length
