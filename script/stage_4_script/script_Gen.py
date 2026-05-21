from local_code.stage_4_code.Dataset_Loader_Generator import Dataset_Loader
from local_code.stage_4_code.Method_Generator import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Generator
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('Generator', '')
    data_obj.dataset_source_folder_path = 'data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    dataset = data_obj.load()
    vs = dataset['vocab']['vocab_size'] 
    method_obj = Method_RNN_Generator('Generator', '',vocab_size=vs)
    method_obj.data=dataset

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_4_result/text_generation'
    result_obj.result_destination_file_name = 'Jokes'

    setting_obj = Setting_Generator('Generator', '')
    evaluate_obj = Evaluate_Accuracy('None', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')
    # ------------------------------------------------------
    # {'accuracy': 0.50504, 'precision': 0.5200343935203657, 'recall': 0.50504, 'f1': 0.39111211141802393}

    