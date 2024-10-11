'''
Created by JaimeVan
2024-10-10
'''
import itertools
import sys
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import jieba
from torchtext.vocab import build_vocab_from_iterator
from gensim.models import Word2Vec

cn_reg = '^[\u4e00-\u9fa5]+$'

def pad_list(input_list, target_length, pad_value=0):
    return input_list[:target_length] + [pad_value] * max(0, target_length - len(input_list))

def label_to_index(label, label_dict):
    label_tensor = torch.zeros(len(label_dict)) 
    label_tensor[label_dict[label]] = 1
    return label_tensor

def split_train_valid(path_data, label_dict, test_size):
    df = pd.read_csv(path_data)
    texts = list(df['text'].values)
    
    labels = df['label'].apply(lambda x: label_to_index(x, label_dict)).values
    X_train, X_valid, y_train, y_valid = train_test_split(texts, labels, test_size=test_size, random_state=42)
    return X_train, X_valid, y_train, y_valid

class pretrainedWordVocabModel():
    def __init__(self, model_path, all_texts, max_words_num = 20000):
        self.max_words_num = max_words_num
        self.load_pretrained_model(model_path)
        
        self.word_index = self.build_word_index(all_texts)
        pass

    def load_pretrained_model(self, model_path):
        embeddings_index = {}
        with open(model_path) as f:
            num, embedding_dim = f.readline().split()

            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        self.embeddings_index = embeddings_index
        self.embedding_dim = int(embedding_dim)
        print('Found %s word vectors, dimension %s' % (len(embeddings_index), embedding_dim))

    def build_word_index(self, text_list):
        tokenized_texts = []
        cn_reg = '^[\u4e00-\u9fa5]+$'
        for text in text_list:
            tokenized_text = [chinese_token for chinese_token in jieba.lcut(text) if re.search(cn_reg, chinese_token)]
            tokenized_texts.append(tokenized_text)

        vocab_set = set()

        for tokens in tokenized_texts:
            vocab_set.update(tokens)

        word_index = {word: idx + 1 for idx, word in enumerate(sorted(vocab_set))}
        word_index["<UNK>"] = len(word_index)
        return word_index

    def get_vocab_from_w2v(self):
        embedding_matrix = np.zeros((self.max_words_num+1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if i < self.max_words_num:
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def get_token_indices(self, texts):
        tokens = []
        for text in texts:
            curr_text_tokens = [token for token in jieba.cut(text) if re.search(cn_reg, token)]
            tokens.append(curr_text_tokens)

        indices = []
        for row in tokens:
            id_ = list(map(lambda x: self.word_index.get(x, self.word_index['<UNK>']), row))
            id_ = pad_list(id_, target_length=64)
            indices.append(id_)
        return np.array(indices)

class myWordVocabModel():
    def __init__(self, all_text, vector_size, window, min_count, workers):

        self.tokens = []
        for text in all_text:
            self.tokens.append([token for token in jieba.cut(text)])

        print("Prepare to Train Word2Vec model")
        self.word2vecModel = Word2Vec(sentences=self.tokens, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        print("Train Word2Vec model OK")

        self.vocab_built = {word: self.word2vecModel.wv[word] for word in self.word2vecModel.wv.index_to_key}
        unk_vector = np.zeros(self.word2vecModel.vector_size)
        self.vocab_built['<UNK>'] = unk_vector

        self.word_to_idx = {word[0]: idx for idx, word in enumerate(self.vocab_built)}
        self.word_to_idx["<UNK>"] = len(self.word_to_idx)

    def sava_word_model(self, path):
        self.word2vecModel.save(path)

    def get_vocab_from_w2v(self):
        return self.vocab_built
    
    def get_token_indices(self, texts):
        tokens = []
        max_token_size = 0
        for text in texts:
            curr_text_tokens = [token for token in jieba.cut(text) if re.search(cn_reg, token)]
            tokens.append(curr_text_tokens)
            if len(curr_text_tokens) >= max_token_size:
                max_token_size = len(curr_text_tokens)

        indices = []
        for row in tokens:
            id_ = list(map(lambda x: self.word_to_idx.get(x, self.word_to_idx['<UNK>']), row))
            id_ = pad_list(id_, target_length=64)
            indices.append(id_)
        return np.array(indices)


class myTextDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def get_text_dataloader(csv_file, label_dict,vector_size=300, test_ratio=0.3, batch_size=32, window=5, min_count=1, workers=4):
    X_train, X_valid, y_train, y_valid = split_train_valid(csv_file, label_dict, test_size=test_ratio)
    all_text = X_train+X_valid
    # VocabModel = myWordVocabModel(all_text, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    VocabModel = pretrainedWordVocabModel('/home/fanjm/PythonPrj/Pytorch-TextCNN/embedding/sgns.zhihu.word', all_text)
    train_ids = VocabModel.get_token_indices(X_train)
    test_ids = VocabModel.get_token_indices(X_valid)

    train_dataset = myTextDataset(train_ids, y_train)
    valid_dataset = myTextDataset(test_ids, y_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader, VocabModel
    
