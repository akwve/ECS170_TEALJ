from local_code.stage_4_code.Dataset_Loader_Classifier import Dataset_Loader
from local_code.stage_4_code.Method_Classifier import Method_RNN_LSTM
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import time

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('Classifier', '')
    data_obj.dataset_source_folder_path = 'data/stage_4_data/text_classification'

    method_obj = Method_RNN_LSTM('Bidirectional LSTM ', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_4_result/LSTM'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('Bidirectional LSTM', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    start = time.time()
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    infodict = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print('Bidirectional LSTM Results for Classification: ' + str(mean_score) + ' +/- ' + str(std_score))
    end = time.time()
    print('************ Finish ************')
    print(f"Time Elapsed : {(end-start):.2f} seconds")
    # ------------------------------------------------------
    # {'accuracy': 0.8512, 'precision': 0.8528314928228127, 'recall': 0.8512, 'f1': 0.8510277881230702}

    