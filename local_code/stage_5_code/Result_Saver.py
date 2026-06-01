'''
Concrete ResultModule class for node classification results
'''

from local_code.base_class.result import result
import os
import pickle


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        file_path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)
        with open(file_path, 'wb') as output_file:
            pickle.dump(self.data, output_file)