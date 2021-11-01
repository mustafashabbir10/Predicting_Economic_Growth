import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext.legacy import data
from torchtext.vocab import Vectors, GloVe
import random


def load_dataset():
    
    TEXT   = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', lower=True)
    LABEL  = data.LabelField()

    fields = {'sentence':('sentence', TEXT), 'sentiment_label':('label', LABEL)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                                        path = "..\Data\DL_Data",
                                                        train = 'train_data.csv',
                                                        test  = 'test_data.csv',
                                                        validation   = 'validation_data.csv',
                                                        format = 'csv',
                                                        fields = fields)
    
    
    MAX_VOCAB_SIZE = 50000

    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)
    
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    
    #print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    
    print(TEXT.vocab.freqs.most_common(20))
    
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                               (train_data, valid_data, test_data), 
                                                               batch_size = BATCH_SIZE,
                                                               sort_key=lambda x: len(x.sentence),
                                                               repeat=True,
                                                               sort=False,
                                                               shuffle=True,
                                                               sort_within_batch=True,
                                                               device = device)
    
    vocab_size = len(TEXT.vocab)
    
    return TEXT, LABEL, vocab_size, train_iterator, valid_iterator, test_iterator
    
    

    
    

