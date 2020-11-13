### Load libraries
##import numpy as np
##from keras.datasets import reuters
##from keras.utils.np_utils import to_categorical
##from keras.preprocessing.text import Tokenizer
##from keras import models
##from keras import layers
##import matplotlib.pyplot as plt
##
### Set random seed
##np.random.seed(0)
##
##
### Set the number of features we want
##number_of_features = 5000
##
### Load feature and target data
##(train_data, train_target_vector), (test_data, test_target_vector) = reuters.load_data(num_words=number_of_features)
##print (train_data)
##print (train_target_vector)
##
### Convert feature data to a one-hot encoded feature matrix
##tokenizer = Tokenizer(num_words=number_of_features)
##train_features = tokenizer.sequences_to_matrix(train_data, mode='binary')
##test_features = tokenizer.sequences_to_matrix(test_data, mode='binary')
### One-hot encode target vector to create a target matrix
##train_target = to_categorical(train_target_vector)
##test_target = to_categorical(test_target_vector)
##
### Start neural network
##network = models.Sequential()
##
### Add fully connected layer with a ReLU activation function
##network.add(layers.Dense(units=100, activation='relu', input_shape=(number_of_features,)))
##
### Add fully connected layer with a ReLU activation function
##network.add(layers.Dense(units=100, activation='relu'))
##
### Add fully connected layer with a softmax activation function
##network.add(layers.Dense(units=46, activation='softmax'))
##
### Compile neural network
##network.compile(loss='categorical_crossentropy', # Cross-entropy
##                optimizer='rmsprop', # Root Mean Square Propagation
##                metrics=['accuracy']) # Accuracy performance metric
##
### Train neural network
##history = network.fit(train_features, # Features
##                      train_target, # Target vector
##                      epochs=3, # Three epochs
##                      verbose=0, # No output
##                      batch_size=100, # Number of observations per batch
##                      validation_data=(test_features, test_target)) # Data to use for evaluation
##
##accr = network.evaluate(test_features,test_target)
##print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
##
##plt.title('Loss')
##plt.plot(history.history['loss'], label='train')
##plt.plot(history.history['val_loss'], label='test')
##plt.legend()
##plt.show();

import logging

from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


## hide warnings##
warnings.filterwarnings('ignore')

## reading the csv file
Dataset = pd.read_csv('data.csv')

daf= {'post':Dataset['Requirement'],'tags':Dataset['Type']}
df = pd.DataFrame(daf)
my_tags = ['O&M','Design','Construction']
df = df[pd.notnull(df['tags'])]
print(df.head(10))
print(df['post'].apply(lambda x: len(x.split(' '))).sum())
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)




train_size = int(len(df) * .75)
train_posts = df['post'][:train_size]
train_tags = df['tags'][:train_size]

test_posts = df['post'][train_size:]
test_tags = df['tags'][train_size:]

max_words = 10000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 15

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

accr = model.evaluate(x_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();






