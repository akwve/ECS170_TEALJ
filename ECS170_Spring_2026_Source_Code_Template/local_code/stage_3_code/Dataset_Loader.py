'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import os
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    #dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        source_path = self.dataset_source_folder_path
        if source_path is None:
            raise ValueError('dataset_source_folder_path is required')

        if not os.path.isabs(source_path):
            source_path = os.path.abspath(source_path)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f'Cannot load stage 3 data from {source_path}')

        # Load pickle file
        with open(source_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Extract train and test data
        train_data = loaded_data['train']
        test_data = loaded_data['test']

        # Convert to lists of images and labels
        X_train = []
        y_train = []
        for instance in train_data:
            X_train.append(instance['image'])
            y_train.append(instance['label'])

        X_test = []
        y_test = []
        for instance in test_data:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int64)

        def to_channel_first(images):
            if images.ndim == 4 and images.shape[-1] in (1, 3):
                images = np.transpose(images, (0, 3, 1, 2))
            elif images.ndim == 3:
                images = images[:, np.newaxis, :, :]
            return images

        X_train = to_channel_first(X_train)
        X_test = to_channel_first(X_test)

        # ORL uses a single-channel CNN; if the source images are RGB, convert to grayscale.
        if 'ORL' in str(source_path).upper() and X_train.ndim == 4 and X_train.shape[1] == 3:
            X_train = X_train.mean(axis=1, keepdims=True)
            X_test = X_test.mean(axis=1, keepdims=True)

        # Normalize pixel values to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        classes = sorted(set(y_train.tolist() + y_test.tolist()))

        label_to_index = {label: index for index, label in enumerate(classes)}
        y_train = np.array([label_to_index[label] for label in y_train.tolist()], dtype=np.int64)
        y_test = np.array([label_to_index[label] for label in y_test.tolist()], dtype=np.int64)

        print(f'Data loaded successfully!')
        print(f'Training set: {X_train.shape}, Labels: {y_train.shape}')
        print(f'Testing set: {X_test.shape}, Labels: {y_test.shape}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'classes': classes
        }