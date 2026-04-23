from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':

    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('MLP dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/'
    result_obj.result_destination_file_name = 'MLP_prediction_result'

    setting_obj = Setting_Train_Test('train test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print(metrics)

    print('************ Finish ************')
    # ------------------------------------------------------
    

    