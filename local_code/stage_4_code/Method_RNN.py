'''
Concrete MethodModule class for RNN (Recurrent Neural Network) implementation
'''

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import time


class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 30
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 64

    # RNN hyperparameters
    vocab_size = None  # Will be inferred from data
    embedding_dim = 128
    hidden_size = 128
    num_layers = 2
    num_classes = None  # Will be inferred from data
    dropout = 0.5

    # it defines the model architecture
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # store loss history for plotting
        self.training_loss_history = []

    def _initialize_architecture(self, vocab_size, num_classes):
        """Initialize RNN architecture based on input dimensions"""
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # Embedding layer to convert word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

        # Use PyTorch's optimized RNN (much faster than manual implementation)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=0.2 if self.num_layers > 1 else 0)

        # Output layers
        self.fc_layer = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout_layer = nn.Dropout(self.dropout)

    def _prepare_input(self, x):
        """Prepare input tensor"""
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        return x.float()

    def forward(self, x):
        '''Forward propagation'''
        x = self._prepare_input(x)

        # Input should be (batch_size, seq_len) with word indices
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        # Convert word indices to embeddings
        # x: (batch_size, seq_len) -> embeddings: (batch_size, seq_len, embedding_dim)
        x = self.embedding(x.long())

        # Use PyTorch's optimized RNN (batch_first=True)
        output, hidden_state = self.rnn(x)

        # Use the last hidden state for classification
        # hidden_state shape: (num_layers, batch_size, hidden_size)
        last_hidden = hidden_state[-1]  # (batch_size, hidden_size)

        # Apply dropout and fully connected layer
        last_hidden = self.dropout_layer(last_hidden)
        y_pred = self.fc_layer(last_hidden)

        return y_pred

    def train_model(self, X, y, vocab_size):
        """Train the RNN model"""
        self.training_loss_history = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Infer vocab_size and number of classes if not set
        if self.vocab_size is None or self.num_classes is None:
            inferred_num_classes = len(np.unique(y))
            self._initialize_architecture(vocab_size, inferred_num_classes)
            self.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        loss_function = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.max_epoch):
            num_samples = len(X)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            total_loss = 0
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = self._prepare_input(X[batch_indices])
                X_batch = X_batch.to(device)
                y_batch = torch.LongTensor(y[batch_indices])
                y_batch = y_batch.to(device)

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            num_batches = int(np.ceil(num_samples / float(self.batch_size)))
            avg_loss = total_loss / max(1, num_batches)
            self.training_loss_history.append(avg_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {avg_loss:.6f}')

            scheduler.step()

    def test(self, X):
        """Test the RNN model with batching to avoid memory issues"""
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            all_predictions = []
            num_samples = len(X)

            # Process in batches to avoid memory overflow
            for i in range(0, num_samples, self.batch_size):
                batch_end = min(i + self.batch_size, num_samples)
                X_batch = self._prepare_input(X[i:batch_end])
                X_batch = X_batch.to(device)

                # Forward pass
                y_pred = self.forward(X_batch)
                # Get predictions
                batch_predictions = y_pred.max(1)[1].cpu().numpy()
                all_predictions.extend(batch_predictions)

            return np.array(all_predictions)

    def run(self):
        """Execute the RNN training and testing pipeline"""
        print('method running...')
        print('--start training...')
        self.method_start_time = time.time()

        # Get vocabulary size from data
        vocab_size = self.data.get('vocab_size', 5000)

        self.train_model(self.data['train']['X'], self.data['train']['y'], vocab_size)
        self.method_training_time = time.time() - self.method_start_time

        print('--start testing...')
        self.method_start_time = time.time()
        pred_y = self.test(self.data['test']['X'])
        self.method_testing_time = time.time() - self.method_start_time
        self.method_stop_time = time.time()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
