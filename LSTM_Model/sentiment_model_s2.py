#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import BertModel, BertTokenizer
# import vsm
from collections import Counter
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch
# from torch_rnn_classifier import TorchRNNClassifier
# from torch_tree_nn import TorchTreeNN
import sst
import utils
from sklearn.preprocessing import LabelEncoder
import warnings
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score
import time
import random
warnings.filterwarnings("ignore")
# pd.set_option('display.max_colwidth',999)
# warnings.filterwarnings("*Truncation was not explicitly*")


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
                            labels      = df.label.to_numpy(),
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
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def labelencoder(train_all, bakeoff_test, sst_dev, dev_all):
    le = LabelEncoder()
    le.fit(train_all['label'].values)
    train_all['label']    =  le.transform(train_all['label'])
    bakeoff_test['label'] = le.transform(bakeoff_test['label'])
    sst_dev['label']      = le.transform(sst_dev['label'])
    dev_all['label']      = le.transform(dev_all['label'])
    
    return le, train_all, bakeoff_test, sst_dev, dev_all
    
def macro_f1(preds, y):
    """
    Returns macro f1 for the batch
    """
    # f1_metric = F1Score('macro')
    top_pred = preds.argmax(1, keepdim = True)
    # print('top pred')
    # print(top_pred)
    # print('actual label')
    # print(y)
    # correct = top_pred.eq(y.view_as(top_pred)).sum()
    macro_f1 = f1_score(y.cpu().data.numpy(),top_pred.cpu().data.numpy(), average='macro')
    # print('macro_f1')
    # print(macro_f1)
    # acc = correct.float() / y.shape[0]
    return macro_f1

def train(model, data_loader, optimizer, criterion, device):
    
    epoch_loss      = 0
    epoch_macro_f1  = 0
    
    model.train()
    
    for batch in data_loader:
        
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        label     = batch['labels'].to(device)

        predictions = model(input_ids).squeeze(1)
        
        loss = criterion(predictions, label)
        
        macro_f1_  = macro_f1(predictions,label)
        # print('Train Batch macro_f1')
        # print(macro_f1_)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss      += loss.item()
        epoch_macro_f1  += macro_f1_.item()
        
    return epoch_loss / len(data_loader), epoch_macro_f1 / len(data_loader)

def evaluate(model, data_loader, criterion,device):
    
    epoch_loss = 0
    epoch_macro_f1  = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for batch in data_loader:

            input_ids = batch['input_ids'].to(device)
            label     = batch['labels'].to(device)
            predictions = model(input_ids).squeeze(1)
            
            loss = criterion(predictions, label)
            
            macro_f1_ = macro_f1(predictions, label)
            # print('Dev Batch macro_f1')
            # print(macro_f1_)

            epoch_loss     += loss.item()
            epoch_macro_f1 += macro_f1_.item()
        
    return epoch_loss / len(data_loader), epoch_macro_f1/ len(data_loader)

def predict_unseen_test(model, unseen_test_dataloader, device):
    model.eval()
    
    sentences   = []
    predictions = []

    with torch.no_grad():
        for d in unseen_test_dataloader:
            batch_sentences = d['sentence']
            input_ids = d['input_ids'].to(device)
            batch_predictions = model(input_ids).squeeze(1).argmax(1, keepdim=True)

            sentences.extend(batch_sentences)
            predictions.extend(batch_predictions)

    predictions   = torch.stack(predictions).cpu()
    predictions_list = list(map(int,list(predictions.numpy().flatten())))
    #print('predictions: ', predictions)
    #print('prediction size: ', predictions.size())
    prediction_df = pd.DataFrame({'sentence':sentences,'prediction':predictions_list})
    return prediction_df

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device %s'%device)
    
    SST_HOME    = os.path.join('data/', 'sentiment')
    sst_train   = sst.train_reader(SST_HOME,include_subtrees=False)
    sst_dev     = sst.dev_reader(SST_HOME)
    bakeoff_dev = sst.bakeoff_dev_reader(SST_HOME)
    
    ### Creating training data
    print('Create training and test')
    bakeoff_train = bakeoff_dev.sample(frac=0.7, random_state=42)
    bakeoff_test_  = bakeoff_dev[~bakeoff_dev.index.isin(bakeoff_train.index)]

    train_all = pd.concat([sst_train, bakeoff_train])
    
    dev_all   = pd.concat([sst_dev, bakeoff_test_])

        #predict on unseen test data
    print('Creating unseen test data')
    bakeoff_test = sst.bakeoff_test_reader(SST_HOME)
    sst_test = sst.test_reader(SST_HOME)
    bakeoff_test['dataset'] = 'bakeoff'
    sst_test['dataset'] = 'sst3'
    test_df = pd.concat((bakeoff_test, sst_test))

    
    #label encoding
    print('label encoding')
    le, train_all, bakeoff_test_, sst_dev, dev_all = labelencoder(train_all, bakeoff_test_, sst_dev, dev_all)
    
    bert_model_weight = 'bert-base-cased'

    #tokenizer to use
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_weight)

    #loading base bert model
    bert_model = BertModel.from_pretrained(bert_model_weight)
    
    # Data prep
    print('Data prep')
    BATCH_SIZE = 32
    MAX_LEN    = 32
    train_data_loader        = create_data_loader(train_all, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    sst_dev_data_loader      = create_data_loader(sst_dev, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    bakeoff_test_data_loader = create_data_loader(bakeoff_test_, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    dev_all_data_loader      = create_data_loader(dev_all, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    unseen_test_dataloader   = create_data_loader(test_df, bert_tokenizer, MAX_LEN, BATCH_SIZE, True)
    
    
    
    #model definition
    print('Defining Model')
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3
    N_LAYERS = 1
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    model = BERTbiLSTM(bert_model,
                       HIDDEN_DIM,
                       OUTPUT_DIM,
                       N_LAYERS,
                       BIDIRECTIONAL,
                       DROPOUT)
    
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    model = model.to(device)
    criterion = criterion.to(device)
    
    #training
    print('Start model training')
    N_EPOCHS = 40
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_f1_mac                          = train(model, train_data_loader, optimizer, criterion, device)
        sst_bakeoff_valid_loss, sst_bakeoff_valid_f1_mac  = evaluate(model, dev_all_data_loader, criterion, device)
#         bakeoff_valid_loss, bakeoff_valid_f1_mac = evaluate(model, bakeoff_test_data_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if sst_bakeoff_valid_loss < best_valid_loss:
            best_valid_loss = sst_bakeoff_valid_loss
            torch.save(model.state_dict(), 'BERT-biLSTM.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_f1_mac*100:.2f}%')
        print(f'\t SST + bakeoff Val. Loss: {sst_bakeoff_valid_loss:.3f} |  SST + bakeoff Val. F1-macro: {sst_bakeoff_valid_f1_mac*100:.2f}%')
#         print(f'\t bakeoff Val. Loss: {bakeoff_valid_loss:.3f} |  bakeoff Val. F1-macro: {bakeoff_valid_f1_mac*100:.2f}%')


    #predict on unseen test data
    print('Predicting on unseen test data using the best model')

    prediction_df = predict_unseen_test(model, unseen_test_dataloader, device)
    prediction_df['prediction'] = list(le.inverse_transform(prediction_df['prediction']))
    
    test_df_out  = pd.merge(prediction_df, test_df, on = ['sentence'])

    test_df_out.to_csv('cs224u-sentiment-bakeoff-entry.csv')
    
    
if __name__== '__main__':
    main()



