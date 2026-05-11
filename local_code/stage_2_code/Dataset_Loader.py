'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    #dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        # Get all .txt files in the folder
        folder_path = self.dataset_source_folder_path
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        print(f'Files found: {files}')

        X_train, y_train = self.load_csv(os.path.join(folder_path, 'train.csv'))
        X_test, y_test = self.load_csv(os.path.join(folder_path, 'test.csv'))

        return {'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}}

    def load_csv(self, path):
        X, y = [], []
        with open(path, 'r') as f:
            next(f)  # skip header (remove if no header)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                elements = [float(i) for i in line.split(',')]
                y.append(elements[0])
                X.append(elements[1:])
        return X, y
