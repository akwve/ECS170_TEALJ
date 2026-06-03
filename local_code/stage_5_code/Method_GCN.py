from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Method_GCN(method, nn.Module):

    data = None
    max_epoch = 200
    learning_rate = 0.01
    patience = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.gc1 = None
        self.gc2 = None
        self.dropout = nn.Dropout(0.5)

        self.to(self.device)

    def forward(self, X, A):
        h = self.gc1(X)
        h = torch.spmm(A, h)
        h = torch.relu(h)
        h = self.dropout(h)

        h = self.gc2(h)
        h = torch.spmm(A, h)
        h = torch.relu(h)
        return h
    def normalize_adj(self,A):
        if A.is_sparse:
            A = A.to_dense()
        I = torch.eye(A.shape[0]).to(A.device)
        A_hat = A + I
        D = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        return D @ A_hat @ D

    def run(self):
        graph = self.data['graph']
        split = self.data['train_test_val']
        X = graph['X'].to(self.device)
        y = graph['y'].to(self.device)
        A = graph['utility']['A'].to(self.device)
        A = self.normalize_adj(A)

        idx_train = split['idx_train']
        idx_val = split['idx_val']
        idx_test = split['idx_test']

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        best_val = 0
        patience_counter = 0
        best_state = None
        loss_curve = []

        for epoch in range(self.max_epoch):
            self.train()
            logits = self.forward(X, A)
            loss = loss_fn(F.log_softmax(logits[idx_train], dim=1), y[idx_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            self.eval()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                val_acc = (pred[idx_val] == y[idx_val]).float().mean().item()

            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val Acc {val_acc:.4f}")
            if val_acc > best_val + 1e-4:
                best_val = val_acc
                patience_counter = 0
                best_state = self.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        with torch.no_grad():
            pred = self.forward(X, A).argmax(dim=1)

        return {
            "pred_y": pred.cpu(),
            "true_y": y.cpu(),
            "loss_curve": loss_curve,
            "best_val_acc": best_val
        }
class Method_GCN_Cora(Method_GCN):
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        self.gc1 = nn.Linear(1433, 16, bias=False)
        self.gc2 = nn.Linear(16, 7, bias=False)
        self.to(self.device)

class Method_GCN_Citeseer(Method_GCN):
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        self.gc1 = nn.Linear(3703, 64, bias=False)
        self.gc2 = nn.Linear(64, 6, bias=False)
        self.to(self.device)
class Method_GCN_Pubmed(Method_GCN):
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        self.gc1 = nn.Linear(500, 64, bias=False)
        self.gc2 = nn.Linear(64, 3, bias=False)
        self.to(self.device)