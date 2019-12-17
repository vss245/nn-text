#based on https://keras.io/examples/lstm_text_generation/
from __future__ import print_function
import tensorflow as tf
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
#note: these are here for now because of bugs
def sample(preds, temperature=1.0): #temperature is the index of surprise in the prediction
    #samples an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
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
                x_pred[0, t, char_idx[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0] #predict
            next_index = sample(preds, diversity)
            next_char = idx_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end) #call our on_epoch_end function to print predicted text
model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
