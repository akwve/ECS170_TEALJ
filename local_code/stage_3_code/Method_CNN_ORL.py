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


class CNN_ORL(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 32

    # Input: 1x112x92
    num_filters_1 = 32
    num_filters_2 = 64
    kernel_size = 5
    pool_size = 2
    padding_size = 1

    # it defines the model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # store loss history for plotting
        self.training_loss_history = []

        # 112x92x3 (only 1 channel is sufficient)
        #Conv2d(in_channels = RGB channel,
        # out_channels = out filter,
        # kernel_size = size of the convolving kernel,
        # stride = control the stride for the cross correlation,
        # padding = control the amount of implcit zero-padding)

        # Input: 1x112x92
        # After conv1: 32x108x88
        # After pool1: 32x54x44
        # After conv2: 64x50x40
        # After pool2: 64x25x20

        self.conv_layer_1 = nn.Conv2d(1, self.num_filters_1, kernel_size=self.kernel_size, padding=self.padding_size//2)
        self.activation_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=self.pool_size)

        self.conv_layer_2 = nn.Conv2d(self.num_filters_1, self.num_filters_2, kernel_size=self.kernel_size, padding=self.padding_size//2)
        self.activation_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=self.pool_size)

        # Flattened size: 64 * 25 * 20 = 32000
        self.fc_layer_1 = nn.Linear(64 * 25 * 20, 128)
        self.activation_3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Output layer:
        self.fc_layer_2 = nn.Linear(128, 40)
        self.activation_4 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''

        # First convolutional block
        x = self.conv_layer_1(x)
        x = self.activation_1(x)
        x = self.pool_1(x)

        # Second convolutional block
        x = self.conv_layer_2(x)
        x = self.activation_2(x)
        x = self.pool_2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layer_1(x)
        x = self.activation_3(x)
        x = self.dropout(x)

        # Output layer
        y_pred = self.activation_4(self.fc_layer_2(x))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        self.training_loss_history = []

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # Mini-batch training
            num_samples = len(X)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            total_loss = 0
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = torch.FloatTensor(X[batch_indices])
                y_batch = torch.LongTensor(y[batch_indices])

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (num_samples // self.batch_size + 1)
            self.training_loss_history.append(avg_loss)

            if epoch % 10 == 0:
                # Calculate training accuracy
                with torch.no_grad():
                    y_pred_all = self.forward(torch.FloatTensor(X))
                    accuracy_evaluator.data = {'true_y': y, 'pred_y': y_pred_all.max(1)[1]}
                    train_metrics = accuracy_evaluator.evaluate()
                    accuracy = train_metrics.get('accuracy', train_metrics.get('accuracy_torchmetrics', 0.0))
                    print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {avg_loss:.6f}')
    
    def test(self, X):
        with torch.no_grad():
            # do the testing, and result the result
            y_pred = self.forward(torch.FloatTensor(X))
            # convert the probability distributions to the corresponding labels
            # instances will get the labels corresponding to the largest probability
            return y_pred.max(1)[1].cpu().numpy()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            