from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN_Cora
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
import time


if __name__ == "__main__":
    np.random.seed(2)
    torch.manual_seed(2)
    data_obj = Dataset_Loader('Node Classification', '')
    data_obj.dataset_source_folder_path = 'data/stage_5_data/cora'
    data_obj.dataset_name = "cora"
    method_obj = Method_GCN_Cora(
        f'GCN-{data_obj.dataset_name}', ''
    )
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_5_result/'
    result_obj.result_destination_file_name = 'cora_result'
    setting_obj = Setting_Train_Test_Split('GCN Setting', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    print("************ Start ************")
    start = time.time()
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print("\n************ Results ************")
    print(metrics)
    end = time.time()
    print(f"Time Elapsed: {end - start:.2f}s")