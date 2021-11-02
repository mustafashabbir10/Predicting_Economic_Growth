import torch
from torchtext.legacy import data
import pandas as pd
import os
import torch.nn as nn
import time
from utils import count_parameters
import torch.optim as optim
import random
from load_data import DataProcessor, Test_DataProcessor, create_data_loader
from RNN import RNN
from utils import train, evaluate, macro_f1, labelencoder, train, evaluate, predict_on_test


def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device %s'%device)
    
    print('Reading Train, Test, Val')
    train_data = pd.read_csv('../Data/Data_DL/train_data.csv')
    valid_data = pd.read_csv('../Data/Data_DL/valid_data.csv')
    test_data = pd.read_csv('../Data/Data_DL/test_data.csv')

    #label encoding
    print('Encoding Labels')
    le, train_data, valid_data, test_data = labelencoder(train_data, valid_data, test_data)
    
    bert_model_weight = 'ProsusAI/finbert'
    
    #tokenizer to use
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_weight)

    #loading base bert model
    bert_model = BertModel.from_pretrained(bert_model_weight)
    
    # Data prep
    print('Data prep')
    BATCH_SIZE = 32
    MAX_LEN    = 32
    train_data_loader  = create_data_loader(train_data, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    valid_data_loader  = create_data_loader(valid_data, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)
    test_data_loader   = create_data_loader(test_data, bert_tokenizer, MAX_LEN, BATCH_SIZE, False)    
    
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

        train_loss, train_f1_mac  = train(model, train_data_loader, optimizer, criterion, device)
        valid_loss, valid_f1_mac  = evaluate(model, valid_data_loader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'FinBERT-biLSTM.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_f1_mac*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. F1-macro: {valid_f1_mac*100:.2f}%')
        
    #predict on test data
    predict_on_test(model, test_data_loader, device)
    
if __name__=='__main__':
    main()
