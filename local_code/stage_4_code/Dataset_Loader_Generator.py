'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from local_code.base_class.dataset import dataset
import csv
import re
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean_text(self,text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'/r/\w+/\S*', '', text)
        text = re.sub(r'\b\w+\.(com|org|net|io|edu)\b', '', text)
        text = text.replace('"', '')
        text = re.sub(r'([?.!,])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load(self):
        print("loading data...")
        jokes = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name,
                  'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    jokes.append(self.clean_text(row[1]))
        tokens = []
        for j in jokes:
            tokens.extend(j.split())
        vocab = sorted(set(tokens))
        vocab = ["<PAD>", "<UNK>", "<END>"] + vocab
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        data = []
        window = 4
        for j in jokes:
            words = j.split()
            idxs = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in words]
            for i in range(len(idxs) - window):
                X = idxs[i:i+3]
                y = idxs[i+3]
                data.append((X, y))
        data = np.array(data, dtype=object)
        np.random.shuffle(data)
        split = int(0.8 * len(data))
        train = data[:split]
        test = data[split:]
        X_train = np.array([x for x, y in train])
        y_train = np.array([y for x, y in train])
        X_test = np.array([x for x, y in test])
        y_test = np.array([y for x, y in test])
        return {"train": {"X": X_train,"y": y_train},"test": {"X": X_test,"y": y_test},"vocab": {"word_to_idx": word_to_idx,"idx_to_word": idx_to_word,"vocab_size": len(vocab)}}