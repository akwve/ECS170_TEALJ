from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    config = {'max_epoch': 50, 'learning_rate': 1e-3, 'batch_size': 64, 'num_filters_1': 32, 'num_filters_2': 64, 'kernel_size': 3}

    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('CIFAR-10', 'CIFAR-10 object recognition dataset')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/CIFAR'

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
    method_obj = CNN_CIFAR('CNN_CIFAR', 'CNN for CIFAR-10 dataset')
    result_obj = Result_Saver('result_saver', '')

    result_obj.result_destination_folder_path = '../../result/stage_3_result/'
    result_obj.result_destination_file_name = 'prediction_result_CIFAR'

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print(f'Configuration: {config}')

    data = data_obj.load()
    method_obj.data = data
    for key, val in config.items():
        setattr(method_obj, key, val)
    method_obj.__init__('CNN_ORL', 'CNN for ORL dataset')
    method_obj.data = data

    result = method_obj.run()

    evaluate_obj.data = result
    metrics = evaluate_obj.evaluate()

    print('************ Overall Performance ************')

    result_obj.data = {
        'pred_y': result['pred_y'],
        'true_y': result['true_y'],
        'accuracy': metrics.get('accuracy', metrics.get('accuracy_torchmetrics', None)),
        'metrics': metrics,
        'configuration': config,
    }
    result_obj.save()

    print([metrics])

    print('************ Finish ************')
    # ------------------------------------------------------
    

    