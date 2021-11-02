import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch
import random


class DataProcessor(Dataset):
    
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences  = sentences
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        
        sentence  = str(self.sentences[item])

        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(sentence,
                                              add_special_tokens    = True,
                                              max_length            = self.max_len,
                                              truncation            = True,
                                              return_token_type_ids = False,
                                              pad_to_max_length     = True,
                                              return_attention_mask = True,
                                              return_tensors        = 'pt')
        
        return {
                'sentence'      : sentence,
                'input_ids'     : encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels'        : torch.tensor(label, dtype=torch.long)
                }


class Test_DataProcessor(Dataset):

    def __init__(self, sentences, tokenizer, max_len):
        self.sentences   = sentences
        self.tokenizer   = tokenizer
        self.max_len     = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):

        sentence  = str(self.sentences[item])

        encoding = self.tokenizer.encode_plus(sentence,
                                              add_special_tokens    = True,
                                              max_length            = self.max_len,
                                              truncation            = True,
                                              return_token_type_ids = False,
                                              pad_to_max_length     = True,
                                              return_attention_mask = True,
                                              return_tensors        = 'pt')

        return {
                'sentence'      : sentence,
                'input_ids'     : encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
                }

    
def create_data_loader(df, tokenizer, max_len, batch_size, istest):
    

    if istest==False:

        ds = DataProcessor(
                            sentences   = df.sentence.to_numpy(),
                            labels      = df.sentiment_label.to_numpy(),
                            tokenizer   = tokenizer,
                            max_len=max_len)
    
        return DataLoader(ds,
                          batch_size=batch_size,
                          num_workers=2
                          )

    else:

        ds = Test_DataProcessor(
                            sentences   = df.sentence.to_numpy(),
                            tokenizer   = tokenizer,
                            max_len=max_len)

        return DataLoader(ds,
                          batch_size=batch_size,
                          num_workers=2
                          )

    

