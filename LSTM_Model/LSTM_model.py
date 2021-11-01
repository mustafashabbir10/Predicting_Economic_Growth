"""
LSTM Model:

"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class BERTbiLSTM(nn.Module):
    
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        #defining bert model
        self.bert = bert
        
        # getting embedding dimesions which will be 768
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        # defining the lstm model
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers    = n_layers,
                            bidirectional = bidirectional,
                            batch_first   = True,
                            dropout       = 0 if n_layers<2 else dropout)
        
        self.out     = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoding):
        
        #text = [batch size, sentence len]
        
        with torch.no_grad():
            embedded = self.bert(encoding)['last_hidden_state']   # [batch_size, sent len, emb dim]

        # print(embedded.size())
        
        outputs, (hidden, cell) = self.lstm(embedded)
        #outputs = [batch_size, sent len, hid dim *2]
        #hidden, cell = [2, batch size, hid_dim]
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        #hidden = [batch_size, hid_dim]
        
        output = self.out(hidden)
        
        return output