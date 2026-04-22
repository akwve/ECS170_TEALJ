'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        def read_csv(file_path):
            X, y = [], []
            with open(file_path, 'r') as f:
                for line in f:
                    elements = line.strip().split(',')

                    # first element = label
                    label = int(elements[0])

                    # remaining 784 = features
                    features = [int(i) for i in elements[1:]]

                    y.append(label)
                    X.append(features)

            return X, y

        train_path = self.dataset_source_folder_path + 'train.csv'
        test_path = self.dataset_source_folder_path + 'test.csv'

        X_train, y_train = read_csv(train_path)
        X_test, y_test = read_csv(test_path)

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }