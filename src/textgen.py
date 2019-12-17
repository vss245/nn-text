#based on https://keras.io/examples/lstm_text_generation/
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import utils
import numpy as np
import random
import sys
import io
#LSTM to generate text
print("loading text...")
data = utils.get_text()
chars = list(set(data)) #unique characters
size = len(chars)
#map between indexes and characters
idx_char = {idx:char for idx, char in enumerate(chars)} #a number for every char
char_idx= {char:idx for idx, char in enumerate(chars)} #a char for every number
#to use keras, data needs to be in the format (nseq, len seq (how much to learn at a time), nfeatures (size))
sentences, next_char, length = utils.prepare()
#sparse representation (create vectors of mostly falses to represent sentences)
x = np.zeros((len(sentences),length,len(chars)), dtype = np.bool)
y = np.zeros((len(sentences),len(chars)), dtype = np.bool)
#we need to get this into the (length, len(chars)) shape to be able to predict upcoming letters
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_idx[char]] = 1
    y[i, char_idx[next_char[i]]] = 1
#start with sequential, add LSTM
print('making a model......')
#we use the whole dataset here to train the model, no test data
model = Sequential() #stack of layers
model.add(LSTM(128, input_shape=(length, len(chars)))) #LSTM layer, 128 memory units
model.add(Dense(len(chars), activation='softmax')) #regular layer, used for outputting a prediction (classification)
#softmax is an activation function, sets threshold
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#measures accuracy, increases as predictions diverge from training
print_callback = LambdaCallback(on_epoch_end=utils.on_epoch_end) #call our on_epoch_end function to print predicted text
model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
