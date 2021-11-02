import time
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def labelencoder(train, valid, test):
    le = LabelEncoder()
    le.fit(train['sentiment_label'].values)
    train['sentiment_label']        =  le.transform(train['sentiment_label'])
    valid['sentiment_label']        = le.transform(valid['sentiment_label'])
    test['sentiment_label']         = le.transform(test['sentiment_label'])
    
    return le, train, valid, test

def macro_f1(preds, y):
    """
    Returns macro f1 for the batch
    """

    top_pred = preds.argmax(1, keepdim = True)

    macro_f1 = f1_score(y.cpu().data.numpy(),top_pred.cpu().data.numpy(), average='macro')

    return macro_f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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

def predict_on_test(model, test_dataloader, device):
    model.eval()
    
    sentences   = []
    labels      = []
    predictions = []

    with torch.no_grad():
        for d in test_dataloader:
            batch_sentences = d['sentence']
            input_ids = d['input_ids'].to(device)
            batch_predictions = model(input_ids).squeeze(1).argmax(1, keepdim=True)

            sentences.extend(batch_sentences)
            predictions.extend(batch_predictions)
            labels.extend(d['labels'])

    predictions   = torch.stack(predictions).cpu()
    predictions_list = list(map(int,list(predictions.numpy().flatten())))
    
    print('Classification report on Test data:')
    print(classification_report(labels, predictions_list))


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

