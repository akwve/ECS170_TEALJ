'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
import torch
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class Evaluate_Accuracy(evaluate):
    data = None
    all_metrics = None

    def _map_labels(self, y_true, y_pred):
        # prefer classes provided by the loader
        classes = None
        if isinstance(self.data, dict):
            classes = self.data.get('classes', None)
        if classes is None:
            labels = np.unique(np.concatenate([np.array(y_true), np.array(y_pred)]))
        else:
            labels = np.array(classes)
        labels = list(labels)
        label_to_idx = {int(l): i for i, l in enumerate(labels)}
        y_true_idx = torch.tensor([label_to_idx[int(x)] for x in y_true], dtype=torch.long)
        y_pred_idx = torch.tensor([label_to_idx[int(x)] for x in y_pred], dtype=torch.long)
        return y_true_idx, y_pred_idx, len(labels), labels
    
    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        # Convert torch tensors to numpy if necessary
        if isinstance(true_y, torch.Tensor):
            true_y = true_y.cpu().numpy()
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()

        y_true_t, y_pred_t, num_classes, labels = self._map_labels(true_y, pred_y)

        results = {}

        acc_metric = MulticlassAccuracy(num_classes=num_classes)
        precision = MulticlassPrecision(num_classes=num_classes, average='weighted')
        recall = MulticlassRecall(num_classes=num_classes, average='weighted')
        f1 = MulticlassF1Score(num_classes=num_classes, average='weighted')


        results['accuracy'] = float(acc_metric(y_pred_t, y_true_t).item())
        results['precision'] = float(precision(y_pred_t, y_true_t).item())
        results['recall'] = float(recall(y_pred_t, y_true_t).item())
        results['f1'] = float(f1(y_pred_t, y_true_t).item())


        return results

        