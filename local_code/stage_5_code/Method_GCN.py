'''
Concrete MethodModule class for GCN node classification
'''

from local_code.base_class.method import method
import copy
import time

import numpy as np
import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.sparse.mm(adj, support) if adj.is_sparse else torch.matmul(adj, support)


class Method_GCN(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 3e-2
    weight_decay = 6e-3
    hidden_dim = 16
    dropout = 0.5
    optimizer = 'adam'
    betas = (0.9, 0.999)
    eps = 1e-6

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.training_loss_history = []
        self.validation_loss_history = []
        self.training_accuracy_history = []
        self.validation_accuracy_history = []
        self.best_state_dict = None
        self.test_loss = None
        self.input_dim = None
        self.num_classes = None
        self.gcn1 = None
        self.gcn2 = None
        self.dropout_layer = nn.Dropout(self.dropout)

    def _initialize_architecture(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gcn1 = GraphConvolution(input_dim, self.hidden_dim)
        self.gcn2 = GraphConvolution(self.hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features, adj):
        x = self.gcn1(features, adj)
        x = torch.relu(x)
        x = self.dropout_layer(x)
        x = self.gcn2(x, adj)
        return x

    def _accuracy(self, logits, labels):
        predictions = logits.argmax(dim=1)
        return (predictions == labels).float().mean().item()

    def train_model(self, graph, idx_train, idx_val):
        self.training_loss_history = []
        self.validation_loss_history = []
        self.training_accuracy_history = []
        self.validation_accuracy_history = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        features = graph['X'].float().to(device)
        labels = graph['y'].long().to(device)
        adj = graph['utility']['A'].to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)

        if self.gcn1 is None or self.gcn2 is None:
            self._initialize_architecture(features.shape[1], int(labels.max().item()) + 1)

        self.to(device)

        opt_name = getattr(self, 'optimizer', 'adam')
        if isinstance(opt_name, str):
            opt_name = opt_name.lower()

        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=getattr(self, 'betas', (0.9, 0.999)),
                eps=getattr(self, 'eps', 1e-6),
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=getattr(self, 'betas', (0.9, 0.999)),
                eps=getattr(self, 'eps', 1e-6),
            )
        loss_function = nn.CrossEntropyLoss()

        best_val_accuracy = -1.0
        best_val_loss = float('inf')
        best_state_dict = None

        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()
            logits = self.forward(features, adj)
            train_logits = logits[idx_train]
            train_labels = labels[idx_train]
            train_loss = loss_function(train_logits, train_labels)
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                logits = self.forward(features, adj)
                train_logits = logits[idx_train]
                val_logits = logits[idx_val]
                train_loss_value = loss_function(train_logits, train_labels).item()
                val_loss_value = loss_function(val_logits, labels[idx_val]).item()
                train_accuracy = self._accuracy(train_logits, train_labels)
                val_accuracy = self._accuracy(val_logits, labels[idx_val])

            self.training_loss_history.append(train_loss_value)
            self.validation_loss_history.append(val_loss_value)
            self.training_accuracy_history.append(train_accuracy)
            self.validation_accuracy_history.append(val_accuracy)

            if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss_value < best_val_loss):
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss_value
                best_state_dict = copy.deepcopy(self.state_dict())

            if epoch % 20 == 0 or epoch == self.max_epoch - 1:
                print(
                    f'Epoch: {epoch:03d} | '
                    f'Train Loss: {train_loss_value:.6f} | '
                    f'Val Loss: {val_loss_value:.6f} | '
                    f'Train Acc: {train_accuracy:.4f} | '
                    f'Val Acc: {val_accuracy:.4f}'
                )

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            self.best_state_dict = best_state_dict

    def test(self, graph, idx_test):
        self.eval()
        device = next(self.parameters()).device
        features = graph['X'].float().to(device)
        labels = graph['y'].long().to(device)
        adj = graph['utility']['A'].to(device)
        idx_test = idx_test.to(device)

        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self.forward(features, adj)
            test_logits = logits[idx_test]
            test_labels = labels[idx_test]
            self.test_loss = loss_function(test_logits, test_labels).item()
            pred_y = test_logits.argmax(dim=1).cpu().numpy()
            true_y = test_labels.cpu().numpy()

        return {'pred_y': pred_y, 'true_y': true_y, 'test_loss': self.test_loss}

    def run(self):
        print('method running...')
        print('--start training...')
        self.method_start_time = time.time()

        graph = self.data['graph']
        train_test_val = self.data['train_test_val']
        self.train_model(graph, train_test_val['idx_train'], train_test_val['idx_val'])
        self.method_training_time = time.time() - self.method_start_time

        print('--start testing...')
        self.method_start_time = time.time()
        result = self.test(graph, train_test_val['idx_test'])
        self.method_testing_time = time.time() - self.method_start_time
        self.method_stop_time = time.time()

        return result
