'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
import os

class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        print('loading results...')
        file_path = os.path.join(self.result_destination_folder_path,
                                 self.result_destination_file_name + '_' + str(self.fold_count))
        f = open(file_path, 'rb')
        self.data = pickle.load(f)
        f.close()