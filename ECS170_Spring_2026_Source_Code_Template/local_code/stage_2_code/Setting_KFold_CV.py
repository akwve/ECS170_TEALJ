'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from local_code.stage_2_code.Training_Convergence_Plot import save_training_loss_curve
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        X_train = np.array(loaded_data['train']['X'])
        y_train = np.array(loaded_data['train']['y'])

        X_test = np.array(loaded_data['test']['X'])
        y_test = np.array(loaded_data['test']['y'])

        kf = KFold(n_splits=self.fold, shuffle=True)

        fold_count = 0
        #score_list = []
        fold_loss_histories = []
        metric_lists = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for train_index, test_index in kf.split(X_train):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_tr, X_tes = X_train[train_index], X_train[test_index]
            y_tr, y_tes = y_train[train_index], y_train[test_index]
        
            # run MethodModule
            self.method.data = {'train': {'X': X_tr, 'y': y_tr}, 'test': {'X': X_tes, 'y': y_tes}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            metrics = self.evaluate.evaluate()
            if not isinstance(metrics, dict):
                score = float(metrics)
                metrics = {
                    'accuracy': score,
                    'precision': score,
                    'recall': score,
                    'f1': score,
                }
            for metric_name in metric_lists:
                metric_lists[metric_name].append(metrics[metric_name])

            if hasattr(self.method, 'training_loss_history') and self.method.training_loss_history:
                fold_loss_histories.append(list(self.method.training_loss_history))

        plot_file_path = self.result.result_destination_folder_path + '/training_loss_curve.png'
        save_training_loss_curve(
            fold_loss_histories,
            plot_file_path,
            'MLP Training Loss Convergence',
            line_color='tab:blue'
        )

        mean_metrics = {metric_name: np.mean(values) for metric_name, values in metric_lists.items()}
        std_metrics = {metric_name: np.std(values) for metric_name, values in metric_lists.items()}

        #score_list.append(self.evaluate.evaluate())
        
        return mean_metrics, std_metrics

        