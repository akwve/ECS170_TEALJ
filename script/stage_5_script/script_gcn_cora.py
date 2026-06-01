import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Training_Convergence_Plot import save_training_loss_curve


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    config = {
        'max_epoch': 300,
        'learning_rate': 3e-2,
        'weight_decay': 6e-3,
        'hidden_dim': 16,
        'dropout': 0.5,
        'optimizer': 'adam',
        'betas': (0.9, 0.999),
        'eps': 1e-6,
    }

    print('Loading dataset...')
    data_obj = Dataset_Loader('cora', 'Cora citation network node classification')
    data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data' / 'stage_5_data' / 'cora')

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
    method_obj = Method_GCN('GCN_Node_Classification_Cora', 'GCN for cora node classification')
    result_obj = Result_Saver('result_saver', '')

    result_folder = PROJECT_ROOT / 'result' / 'stage_5_result'
    result_obj.result_destination_folder_path = str(result_folder)
    result_obj.result_destination_file_name = 'prediction_result_cora'

    print('************ Start ************')
    print('Dataset: cora')
    print(f'Configuration: {config}')

    data = data_obj.load()
    method_obj.data = data
    for key, value in config.items():
        setattr(method_obj, key, value)

    print('Running method...')
    result = method_obj.run()

    evaluate_obj.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
    metrics = evaluate_obj.evaluate()

    print('\n========== Test Results ==========' )
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1:              {metrics['f1']:.4f}")
    print(f"Test Loss:       {result['test_loss']:.6f}")
    print('==================================\n')

    plot_file_path = str(result_folder / 'gcn_learning_curves_cora.png')
    save_training_loss_curve([method_obj.training_loss_history], plot_file_path, 'GCN Training Loss on Cora')

    result_obj.data = {
        'pred_y': result['pred_y'],
        'true_y': result['true_y'],
        'test_loss': result['test_loss'],
        'metrics': metrics,
        'configuration': config,
        'dataset': 'cora',
        'training_loss_history': method_obj.training_loss_history,
        'method_training_time': method_obj.method_training_time,
        'method_testing_time': method_obj.method_testing_time,
    }
    result_obj.save()

    print('************ Finish ************')