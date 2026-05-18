import os
import sys
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Training_Convergence_Plot import save_training_loss_curve
import numpy as np
import torch

# ---- RNN Text Classification script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    config = {
        'max_epoch': 10     ,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'max_sequence_length': 500,
        'min_word_frequency': 5
    }

    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    print('Loading dataset...')
    data_obj = Dataset_Loader('IMDb', 'IMDb movie reviews sentiment classification')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
    method_obj = Method_RNN('RNN_Text_Classification', 'RNN for text classification')

    # Apply configuration
    for key, val in config.items():
        if hasattr(data_obj, key):
            setattr(data_obj, key, val)
        if hasattr(method_obj, key):
            setattr(method_obj, key, val)

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print(f'Configuration: {config}')

    # Load data
    data = data_obj.load()
    method_obj.data = data

    # Apply final configuration
    for key, val in config.items():
        if hasattr(method_obj, key):
            setattr(method_obj, key, val)

    # Train and test
    print('Running method...')
    result = method_obj.run()

    # Evaluate
    evaluate_obj.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
    metrics = evaluate_obj.evaluate()
    print(f'\n========== Test Results ==========')
    print(f'Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print(f'F1 Score:  {metrics["f1"]:.4f}')
    print(f'==================================\n')

    # Plot training loss
    plot_file_path = '../../result/stage_4_result/training_loss_curve_rnn.png'
    os.makedirs(os.path.dirname(plot_file_path), exist_ok=True)

    if method_obj.training_loss_history:
        save_training_loss_curve(
            [method_obj.training_loss_history],
            plot_file_path,
            'RNN Training Loss Convergence',
            line_color='tab:blue'
        )
        print(f'Training loss curve saved to {plot_file_path}')

    print('************ End ************')
