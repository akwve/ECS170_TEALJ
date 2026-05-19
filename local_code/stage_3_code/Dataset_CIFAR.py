'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
import random
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def augment(self,img):
        if random.random() < 0.5:
            img = np.flip(img,axis=2)
        pad = 4
        img = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        x = random.randint(0, 8)
        y = random.randint(0, 8)
        img = img[:, x:x+32, y:y+32]
        return img

    def load(self):        
        print('loading data...')
        fpath = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        # pickle is already loaded, now what we can do it translate it into the regular split
        X_train=[]
        Y_train=[]
        for i in data['train']:
            img = np.array(i['image'], dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            X_train.append(self.augment(img))
            Y_train.append(i['label'])
        X_test=[]
        Y_test=[]
        for j in data['test']:
            img = np.array(j['image'], dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            X_test.append(img)
            Y_test.append(j['label'])

        return {'train':{'X':X_train, 'y':Y_train}, 'test':{'X':X_test,'y':Y_test}}