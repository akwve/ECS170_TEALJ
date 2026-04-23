'''
Concrete MethodModule class for a specific learning MethodModule
'''
from sklearn.metrics import accuracy_score

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        # initialize layers after seeing data
        self.model = None

    def build_model(self, input_dim, num_classes):

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def train_model(self, X, y):
        print("training model...")

        X = np.array(X) / 255.0
        y = np.array(y)

        input_dim = X.shape[1]
        num_classes = len(np.unique(y))

        self.build_model(input_dim, num_classes)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        loss_list = []
        acc_list = []

        for epoch in range(self.max_epoch):
            self.train()

            # forward
            y_pred = self.forward(X_tensor)

            # loss
            loss = loss_function(y_pred, y_tensor)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            preds = y_pred.argmax(dim = 1).detach().numpy()
            true = y_tensor.numpy()

            acc = accuracy_score(true, preds)

            loss_list.append(loss.item())
            acc_list.append(acc)

            if epoch % 50 == 0:
                print('epoch:', epoch, 'loss:', loss.item(), 'accuracy:', acc)

        plt.figure()
        plt.plot(loss_list, label = 'Loss')
        plt.plot(acc_list, label = 'Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    def test(self, X):
        print("testing model...")

        X_tensor = torch.FloatTensor(np.array(X) / 255.0)

        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.argmax(dim = 1).numpy()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            