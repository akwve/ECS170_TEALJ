
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class Evaluate_Accuracy(evaluate):
    data = None
    all_metrics = None

    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        # ---- convert torch → numpy if needed ----
        if hasattr(y_true, "cpu"):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu().numpy()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # ---- OPTIONAL: apply mask (THIS IS THE FIX) ----
        # expected: self.data['mask'] = idx_test (or idx_val)
        if 'mask' in self.data and self.data['mask'] is not None:
            mask = self.data['mask']

            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()

            mask = np.array(mask)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        # ---- safety check ----
        assert len(y_true) == len(y_pred), "Shape mismatch in evaluation"

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        self.all_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

        return self.all_metrics