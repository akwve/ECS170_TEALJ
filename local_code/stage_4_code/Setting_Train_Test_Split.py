'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        # run MethodModule
        self.method.data = self.dataset.load()
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None


class Setting_Generator(setting):
    fold = 3

    def load_run_save_evaluate(self):

        self.method.data = self.dataset.load()

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        print('\nGenerated Jokes:\n')

        for i, joke in enumerate(learned_result['generated_text']):
            print(f'{i+1}. {joke}\n')

        return learned_result, None