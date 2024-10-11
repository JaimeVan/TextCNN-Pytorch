'''
Created by JaimeVan
2024-10-09
'''
import itertools

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import jieba

class TfidfVectorizerCustom():
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, documents):
        all_single_text = [text.split() for text in documents]
        all_single_text = list(itertools.chain.from_iterable(all_single_text))
        self.vectorizer.fit(all_single_text)

    def transform(self, text):
        documents = text.split()
        tfidf_matrix = self.vectorizer.transform(documents).toarray()

        return tfidf_matrix
    
    def get_vocab(self):
        return self.vectorizer.vocabulary_

class TfidfTextDataset(Dataset):
    def __init__(self, texts, labels, tfidf, max_features):
        self.texts = texts
        self.Tfidf_tokenizer = tfidf
        self.labels = labels
        self.max_features = max_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features_vector = torch.zeros(30, self.max_features, dtype=torch.long)
        text_vector = torch.tensor(self.Tfidf_tokenizer.transform(self.texts[idx]), dtype=torch.long)

        features_vector[:text_vector.shape[0], :] = text_vector

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features_vector, label
    

def label_to_index(label, label_dict):
    label_tensor = torch.zeros(len(label_dict)) 
    label_tensor[label_dict[label]] = 1
    return label_tensor

def split_train_valid(path_data, label_dict, test_size=0.3):
    df = pd.read_csv(path_data)
    texts = list(df['text'].values)
    # texts = [' '.join(jieba.cut(text)) for text in texts]
    labels = df['label'].apply(lambda x: label_to_index(x, label_dict)).values
    X_train, X_valid, y_train, y_valid = train_test_split(texts, labels, test_size=test_size, random_state=42)
    return X_train, X_valid, y_train, y_valid

def create_dataloader(X_train, y_train, X_valid, y_valid, batch_size=32, max_features=1000):
    myTfidf_text = TfidfVectorizerCustom(max_features=max_features)
    myTfidf_text.fit(X_train+X_valid)

    train_dataset = TfidfTextDataset(X_train, y_train, tfidf=myTfidf_text, max_features=max_features)
    valid_dataset = TfidfTextDataset(X_valid, y_valid, tfidf=myTfidf_text, max_features=max_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, myTfidf_text.get_vocab()

X_train, X_valid, y_train, y_valid = split_train_valid('/home/fanjm/PythonPrj/Pytorch-TextCNN/data/smp2017.csv')