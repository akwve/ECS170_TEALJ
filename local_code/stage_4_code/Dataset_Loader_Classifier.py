'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os
import re
from collections import Counter
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download once
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.word2idx = {'<PAD>':0, '<UNK>':1}

        self.max_length = 200
        self.vocab_size = 20000

    def clean(self,text):
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        cleaned = []
        for token in tokens:
            if token not in stop_words:
                token = lemmatizer.lemmatize(token)
                cleaned.append(token)
        return ' '.join(cleaned)
    
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for idx, (word, _) in enumerate(
            counter.most_common(self.vocab_size - 2),
            start=2
        ):
            self.word2idx[word] = idx

    def encode(self, text):
        seq = [
            self.word2idx.get(token, 1)
            for token in text.split()
        ]
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        else:
            seq += [0] * (self.max_length - len(seq))
        return seq

    def load_split(self, split):
        X = []
        y = []
        split_path = os.path.join(
            self.dataset_source_folder_path,
            split
        )
        for label_name, label in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(split_path, label_name)
            for fname in os.listdir(folder):
                if fname.endswith('.txt'):
                    fpath = os.path.join(folder, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        text = self.clean(f.read())
                    X.append(text)
                    y.append(label)
        return X, y

    def load(self):
        print("Starting to load Data")
        X_train, y_train = self.load_split('train')

        X_test, y_test = self.load_split('test')

        self.build_vocab(X_train)

        X_train = [self.encode(x) for x in X_train]

        X_test = [self.encode(x) for x in X_test]
        print("Done Loading Data")
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }