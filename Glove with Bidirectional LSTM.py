#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

with open('/Users/favolithomas/Desktop/CS/NLP/data.txt', 'r') as f:

    # initialize three empty lists to store the separated data
    col1 = []
    col2 = []
    col3 = []

    # read each line of the file
    for line in f:
        # split the line at each '|' character
        data = line.strip().split('|')
        # append each column of data to its respective list
        col1.append(data[0])
        col2.append(data[1])
        col3.append(data[2])

df = pd.DataFrame({'sentence': col2, 'labels': col3})

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['sentence'])
sequences = tokenizer.texts_to_sequences(df['sentence'])

# Pad the sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# One-hot encode the labels
labels = pd.get_dummies(df['labels']).values

# Load the pre-trained GloVe embeddings
embedding_dict = {}
with open('/Users/favolithomas/Desktop/CS/NLP/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = vector

# Create an embedding matrix for the words in the dataset
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model architecture
model = Sequential()
model.add(Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(labels.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Train and test the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
