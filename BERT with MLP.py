#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the data
with open('/Users/favolithomas/Desktop/CS/NLP/data.txt', 'r') as f:
    col1 = []
    col2 = []
    col3 = []
    for line in f:
        data = line.strip().split('|')
        col1.append(data[0])
        col2.append(data[1])
        col3.append(data[2])
df = pd.DataFrame({'sentence': col2, 'label': col3})

# Tokenize the sentences and generate embeddings using BERT
input_ids = []
attention_masks = []
for sent in df['sentence']:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 64,           
                        pad_to_max_length = True,
                        return_attention_mask = True,  
                        return_tensors = 'pt'
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_masks)

# Encode the labels
le = preprocessing.LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])

# Convert the encoded sentences to features and labels
features = last_hidden_states[0][:, 0, :].numpy()
labels = df['label'].values

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors and create data loaders
train_features = torch.tensor(train_features)
test_features = torch.tensor(test_features)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_data = TensorDataset(train_features, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
test_data = TensorDataset(test_features, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Instantiate the MLP model, optimizer, and loss function
model = MLP(input_size=768, hidden_size=256, output_size=12)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(data.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, running_loss/len(train_dataloader)))

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_dataloader:
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy on test set: {:.2f}%'.format(100 * correct / total))