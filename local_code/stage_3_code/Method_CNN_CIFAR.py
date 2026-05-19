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


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 30
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
        self.conv1 = nn.Conv2d(3,32,kernel_size= 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32,32,kernel_size= 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        # Double the convolution layers before each pooling compared to ORL
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fcmini = nn.Linear(128*8*8,512)
        self.fc1 = nn.Linear(64*8*8,512)
        self.relu5 = nn.ReLU()

        self.drop = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512,10)
        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.pool1(h)

        h = self.relu3(self.bn3(self.conv3(h)))
        h = self.relu4(self.bn4(self.conv4(h)))
        h = self.pool2(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)

        h = self.relu5(h)
        h = self.drop(h)
        y_pred = self.fc2(h)

        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
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
                X_batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
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
        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1].detach().cpu()
        
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            