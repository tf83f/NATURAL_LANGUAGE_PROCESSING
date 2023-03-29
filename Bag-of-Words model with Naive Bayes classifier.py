#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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
        
# combine col2 and col3 into a list of tuples
data = [(x, y) for x, y in zip(col2, col3)]

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# extract features using the Bag-of-Words model
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform([x for x, y in train_data])
test_features = vectorizer.transform([x for x, y in test_data])

# train a Naïve Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_features, [y for x, y in train_data])

# evaluate the classifier on the test set
accuracy = classifier.score(test_features, [y for x, y in test_data])
print("Accuracy:", accuracy)

# get the predicted labels for the test data
predicted_labels = classifier.predict(test_features)

# get the unique labels
labels = np.unique([y for x, y in test_data])

# compute the accuracy for each label
accuracy = []
for label in labels:
    idx = [i for i, x in enumerate(test_data) if x[1] == label]
    acc = classifier.score(test_features[idx], [test_data[i][1] for i in idx])
    accuracy.append(acc)

# plot the accuracy for each label
plt.bar(labels, accuracy)
plt.title("Dialogue Act Classification Accuracy")
plt.xlabel("Dialogue Act Label")
plt.xticks(rotation=90) 
plt.ylabel("Accuracy")
plt.show()

# create a dictionary to count the frequency of each label
label_freq = {}
for label in col3:
    if label in label_freq:
        label_freq[label] += 1
    else:
        label_freq[label] = 1

# create a bar chart of the label frequencies
plt.bar(label_freq.keys(), label_freq.values())
plt.title("Frequency of each Dialogue Act label")
plt.xlabel("Dialogue Act Label")
plt.xticks(rotation=90) 
plt.ylabel("Frequency")
plt.show()