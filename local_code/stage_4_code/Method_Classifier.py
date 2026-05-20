'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_RNN_bidirectional(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 11
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        self.embedding = nn.Embedding(20000, 128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128,batch_first=True,bidirectional=True)

        self.norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        h = self.embedding(x)
        output, (hidden, cell) = self.lstm(h)
        h = output.mean(dim=1)
        h = self.norm(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.drop(h)

        return self.fc2(h)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # full batch is too much ram, so it is limited to batches of 64 so it's doable by the GPU
        batch_size = 64
        X = np.array(X)
        y = np.array(y)
        for epoch in range(self.max_epoch):
            epoch_loss = 0
            all_preds = []
            all_true = []
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            for i in range(0, len(X), batch_size):
                X_batch = torch.LongTensor(X[i:i+batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i:i+batch_size]).to(self.device)
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * X_batch.size(0)
                all_preds.append(y_pred.max(1)[1].detach().cpu())
                all_true.append(y_batch.detach().cpu())
            
            accuracy_evaluator.data = {
                'true_y': torch.cat(all_true),
                'pred_y': torch.cat(all_preds)
            }
            epoch_loss = epoch_loss / len(X)
            print('Epoch:', epoch,'Accuracy:', accuracy_evaluator.evaluate(),'Loss:', epoch_loss)
    
    def test(self, X):
        self.eval()
        batch_size = 64
        preds = []
        X = np.array(X, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.LongTensor(X[i:i+batch_size]).to(self.device)
                y_pred = self.forward(X_batch)
                preds.append(torch.argmax(y_pred, dim=1).cpu())
        return torch.cat(preds)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}


class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.embedding = nn.Embedding(20000, 128, padding_idx= 0)
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64,2)
        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        h = self.embedding(x)
        output, (hidden, cell) = self.lstm(h)
        h = hidden[-1]
        return self.fc(h)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # full batch is too much ram, so it is limited to batches of 64 so it's doable by the GPU
        batch_size = 64
        X = np.array(X)
        y = np.array(y)
        for epoch in range(self.max_epoch):
            epoch_loss = 0
            all_preds = []
            all_true = []
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            for i in range(0, len(X), batch_size):
                X_batch = torch.LongTensor(X[i:i+batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i:i+batch_size]).to(self.device)
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * X_batch.size(0)
                all_preds.append(y_pred.max(1)[1].detach().cpu())
                all_true.append(y_batch.detach().cpu())
            
            accuracy_evaluator.data = {
                'true_y': torch.cat(all_true),
                'pred_y': torch.cat(all_preds)
            }
            epoch_loss = epoch_loss / len(X)
            print('Epoch:', epoch,'Accuracy:', accuracy_evaluator.evaluate(),'Loss:', epoch_loss)
    
    def test(self, X):
        self.eval()
        batch_size = 64
        preds = []
        X = np.array(X, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.LongTensor(X[i:i+batch_size]).to(self.device)
                y_pred = self.forward(X_batch)
                preds.append(torch.argmax(y_pred, dim=1).cpu())
        return torch.cat(preds)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}