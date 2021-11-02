from utils import labelencoder, predict_unseen_test
from load_data import Test_DataProcessor, create_data_loader
import torch
from LSTM_model import BERTbiLSTM
from transformers import BertModel, BertTokenizer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_model_weight = 'ProsusAI/finbert'

    #tokenizer to use
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_weight)

    #loading base bert model
    bert_model = BertModel.from_pretrained(bert_model_weight)
    
    ## Model params for FinBERT-biLSTM_2.pt for FinBERT-biLSTM.pt change HIDDEN_DIM TO 128 and N_LAYERS TO 1
    print('Load the Model')
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    model = BERTbiLSTM(bert_model, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    model.load_state_dict(torch.load('FinBERT-biLSTM_2.pt'))
    model.to(device)
    
    print('Load news articles')
    unlabeled_news_df = pd.read_csv('sample_test.csv')   ### CHANGE ME
    
    print('Data prep')
    BATCH_SIZE = 32
    MAX_LEN    = 32
    unseen_test_dataloader   = create_data_loader(unlabeled_news_df, bert_tokenizer, MAX_LEN, BATCH_SIZE, True)
    
    print('Predicting on Unseen Test data')
    
    predicted_news_sentiment = predict_unseen_test(model, unseen_test_dataloader, device)
    
    ## saving the predictions
    predicted_news_sentiment.to_csv('Predicted_news_df.csv', index=False)
    
    
if __name__=='__main__':
    main()