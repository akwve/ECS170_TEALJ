from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Training_Convergence_Plot import save_training_loss_curve
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    config = {'max_epoch': 200, 'learning_rate': 1e-3, 'batch_size': 32, 'num_filters_1': 32, 'num_filters_2': 64, 'kernel_size': 5, 'padding_size': 1}

    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('ORL', 'ORL face recognition dataset')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/ORL'

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
    method_obj = CNN_ORL('CNN_ORL', 'CNN for ORL dataset')
    result_obj = Result_Saver('result_saver', '')

    result_obj.result_destination_folder_path = '../../result/stage_3_result/'
    result_obj.result_destination_file_name = 'prediction_result_ORL'

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


    plot_file_path = '../../result/stage_3_result/training_loss_curve_orl.png'
    save_training_loss_curve(
        [method_obj.training_loss_history],
        plot_file_path,
        'ORL Training Loss Convergence',
        line_color='tab:blue'
    )


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
    

    