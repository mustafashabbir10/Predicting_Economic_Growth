import torch
from torchtext.legacy import data
import pandas as pd
import os
import torch.nn as nn
import time
from utils import count_parameters
import torch.optim as optim
import random
from load_data import load_dataset
from RNN import RNN
from utils import train, evaluate, macro_f1


def main():

    SEED= 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load data
    TEXT, LABEL, vocab_size,  train_iter, valid_iter, test_iter = load_dataset()
    
    print('Training Size: %d'%len(train_iter.dataset.examples))
    print('Validation Size: %d'%len(valid_iter.dataset.examples))
    print('Test Size: %d'%len(test_iter.dataset.examples))

    train_label_count_dict = {'positive':0, 'negative':0, 'neutral':0}
    for lab in train_iter.dataset.examples:
        train_label_count_dict[lab.label]+=1
    print('Training Label count:')
    print(train_label_count_dict)

    print('###############')
    val_label_count_dict = {'positive':0, 'negative':0, 'neutral':0}
    for lab in valid_iter.dataset.examples:
        val_label_count_dict[lab.label]+=1
    print('Validation Label count:')
    print(val_label_count_dict)

    print('###############')
    test_label_count_dict = {'positive':0, 'negative':0, 'neutral':0}
    for lab in test_iter.dataset.examples:
        test_label_count_dict[lab.label]+=1
    print('Test Label count:')
    print(test_label_count_dict)


    ### RNN Model
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)


    print(f'The model has {count_parameters(model):,} trainable parameters')
    

    N_EPOCHS = 5

    best_valid_loss = float('inf')
    print('Start training')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_macro_f1 = train(model, train_iter, optimizer, criterion, macro_f1)
        valid_loss, valid_macro_f1 = evaluate(model, valid_iter, criterion, macro_f1)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'rnn1-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_macro_f1*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_macro_f1*100:.2f}%')



if __name__=='__main__':
    main()
