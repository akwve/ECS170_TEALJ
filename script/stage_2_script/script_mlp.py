from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
<<<<<<< HEAD
=======
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
>>>>>>> 3a2c4d3a75bc06254e50e7157406749cfbec2287
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
<<<<<<< HEAD
    data_obj = Dataset_Loader('test and train', '')
    data_obj.dataset_source_folder_path = 'data/stage_2_data/'
    data_obj.dataset_source_file_name = ['test.csv', 'train.csv']
=======
    data_obj = Dataset_Loader('stage2', '')
    data_obj.dataset_source_folder_path = 'J:/My Drive/ECS 170/stage_2_data/'
    #data_obj.dataset_source_file_name = 'toy_data_file.txt'
>>>>>>> 3a2c4d3a75bc06254e50e7157406749cfbec2287

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
<<<<<<< HEAD
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
=======
    result_obj.result_destination_folder_path = '../../result/stage_2_result'
    result_obj.result_destination_file_name = 'MLP_prediction_result'

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
>>>>>>> 3a2c4d3a75bc06254e50e7157406749cfbec2287
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
<<<<<<< HEAD
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
=======
    mean_metrics, std_metrics = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')

    print('MLP Accuracy: ' + str(mean_metrics['accuracy']) + ' +/- ' + str(std_metrics['accuracy']))
    print('MLP Precision: ' + str(mean_metrics['precision']) + ' +/- ' + str(std_metrics['precision']))
    print('MLP Recall: ' + str(mean_metrics['recall']) + ' +/- ' + str(std_metrics['recall']))
    print('MLP F1: ' + str(mean_metrics['f1']) + ' +/- ' + str(std_metrics['f1']))

>>>>>>> 3a2c4d3a75bc06254e50e7157406749cfbec2287
    print('************ Finish ************')
    # ------------------------------------------------------
    

    