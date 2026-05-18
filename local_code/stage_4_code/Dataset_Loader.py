'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import os
import re
import string
from collections import Counter


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    # Text preprocessing parameters
    max_sequence_length = 500
    min_word_frequency = 5

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.vocabulary = {}
        self.reverse_vocabulary = {}

    def _clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-z]+;', '', text)

        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _tokenize(self, text):
        """Tokenize text into words"""
        tokens = text.split()
        return tokens

    def _build_vocabulary(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()

        # Count word frequencies
        for text in texts:
            cleaned = self._clean_text(text)
            tokens = self._tokenize(cleaned)
            word_counts.update(tokens)

        # Filter words by minimum frequency and build vocabulary
        # Reserve index 0 for padding, 1 for unknown words
        self.vocabulary = {'<PAD>': 0, '<UNK>': 1}
        vocab_index = 2

        for word, count in word_counts.most_common():
            if count >= self.min_word_frequency:
                self.vocabulary[word] = vocab_index
                vocab_index += 1

        # Build reverse vocabulary
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

        print(f'Vocabulary built with {len(self.vocabulary)} unique words')
        return self.vocabulary

    def _text_to_sequence(self, text):
        """Convert text to sequence of word indices"""
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)

        # Convert tokens to indices
        sequence = []
        for token in tokens:
            if token in self.vocabulary:
                sequence.append(self.vocabulary[token])
            else:
                sequence.append(self.vocabulary['<UNK>'])

        # Pad or truncate sequence
        if len(sequence) < self.max_sequence_length:
            sequence = sequence + [self.vocabulary['<PAD>']] * (self.max_sequence_length - len(sequence))
        else:
            sequence = sequence[:self.max_sequence_length]

        return np.array(sequence, dtype=np.int64)

    def _load_texts_from_directory(self, directory_path, label):
        """Load all text files from a directory"""
        texts = []
        labels = []

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f'Directory not found: {directory_path}')

        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append(label)
                except Exception as e:
                    print(f'Warning: Could not read {filepath}: {e}')

        return texts, labels

    def load(self):
        """Load and preprocess the text classification dataset"""
        print('loading data...')

        source_path = self.dataset_source_folder_path
        if source_path is None:
            raise ValueError('dataset_source_folder_path is required')

        if not os.path.isabs(source_path):
            source_path = os.path.abspath(source_path)

        if not os.path.isdir(source_path):
            raise FileNotFoundError(f'Cannot load stage 4 data from {source_path}')

        # Load training data
        train_pos_path = os.path.join(source_path, 'train', 'pos')
        train_neg_path = os.path.join(source_path, 'train', 'neg')

        print('Loading training data...')
        train_texts_pos, train_labels_pos = self._load_texts_from_directory(train_pos_path, label=1)
        train_texts_neg, train_labels_neg = self._load_texts_from_directory(train_neg_path, label=0)

        train_texts = train_texts_pos + train_texts_neg
        train_labels = train_labels_pos + train_labels_neg

        print(f'Loaded {len(train_texts)} training samples')

        # Load test data
        test_pos_path = os.path.join(source_path, 'test', 'pos')
        test_neg_path = os.path.join(source_path, 'test', 'neg')

        print('Loading test data...')
        test_texts_pos, test_labels_pos = self._load_texts_from_directory(test_pos_path, label=1)
        test_texts_neg, test_labels_neg = self._load_texts_from_directory(test_neg_path, label=0)

        test_texts = test_texts_pos + test_texts_neg
        test_labels = test_labels_pos + test_labels_neg

        print(f'Loaded {len(test_texts)} test samples')

        # Build vocabulary from training texts
        print('Building vocabulary...')
        self._build_vocabulary(train_texts)

        # Convert texts to sequences
        print('Converting texts to sequences...')
        X_train = np.array([self._text_to_sequence(text) for text in train_texts], dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.int64)

        X_test = np.array([self._text_to_sequence(text) for text in test_texts], dtype=np.float32)
        y_test = np.array(test_labels, dtype=np.int64)

        print(f'Data loaded successfully!')
        print(f'Training set: {X_train.shape}, Labels: {y_train.shape}')
        print(f'Testing set: {X_test.shape}, Labels: {y_test.shape}')
        print(f'Class distribution - Train: {np.bincount(y_train)}')
        print(f'Class distribution - Test: {np.bincount(y_test)}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'vocabulary': self.vocabulary,
            'vocab_size': len(self.vocabulary),
            'classes': [0, 1]
        }
