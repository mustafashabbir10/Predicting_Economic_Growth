import time
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn as nn
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def macro_f1(preds, y):
    """
    Returns macro f1 for the batch
    """
    top_pred = preds.argmax(1, keepdim = True)

    macro_f1 = f1_score(y.cpu().data.numpy(),top_pred.cpu().data.numpy(), average='macro')
#     macro_f1 = f1_score(y.cpu().data.numpy(),top_pred.cpu().data.numpy(), average='macro')


    return macro_f1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, macro_f1):
    
    epoch_loss = 0
    epoch_macro_f1 = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.sentence).squeeze(1)
        
        loss     = criterion(predictions, batch.label)
                
        macro_f1_sc = macro_f1(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_macro_f1  += macro_f1_sc.item()
        
    return epoch_loss / len(iterator), epoch_macro_f1 / len(iterator)


def evaluate(model, iterator, criterion, macro_f1):
    
    epoch_loss = 0
    epoch_macro_f1 = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.sentence).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            macro_f1_sc = macro_f1(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_macro_f1 += macro_f1_sc.item()
        
    return epoch_loss / len(iterator), epoch_macro_f1 / len(iterator)