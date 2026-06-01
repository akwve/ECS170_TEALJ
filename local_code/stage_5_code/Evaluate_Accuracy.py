'''
Concrete Evaluate class for multi-class node classification metrics
'''

from local_code.base_class.evaluate import evaluate
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def __init__(self, eName=None, eDescription=None):
        super().__init__(eName, eDescription)

    def evaluate(self):
        if self.data is None:
            raise ValueError('data is required for evaluation')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        if isinstance(true_y, torch.Tensor):
            true_y = true_y.cpu().numpy()
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()

        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='macro', zero_division=0)
        recall = recall_score(true_y, pred_y, average='macro', zero_division=0)
        f1 = f1_score(true_y, pred_y, average='macro', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
