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

def sample(preds, temperature=1.0): #temperature is the index of surprise in the prediction
    #samples an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    data = get_text()
    length = 50;
    #this happens at the end of every epoch (cycle)
    print()
    print('----- generating text after epoch: %d' % epoch)

    start_index = random.randint(0, len(data) - length - 1) #sets random vars for the seed
    for diversity in [0.2, 0.5, 1.0, 1.2]: #prints out predicted text at 5 different temperatures
        print('----- diversity:', diversity)

        generated = ''
        sentence = data[start_index: start_index + length]
        generated += sentence
        print('----- generating with seed: "' + sentence + '"') #random sentence to start from
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, length, len(chars))) #vector for predictions
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0] #predict
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
