'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    #dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        Xtrain = []
        ytrain = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name[0], 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            Xtrain.append(elements[:-1])
            ytrain.append(elements[-1])
        f.close()
        Xtest = []
        ytest = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name[1], 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            Xtest.append(elements[:-1])
            ytest.append(elements[-1])
        f.close()
        return {'train':{'X': Xtrain, 'y': ytrain}, 'test':{'X': Xtest, 'y': ytest}}
