import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from Shallow_ML_Models.DataPreprocessor import TextPreprocessor
from Shallow_ML_Models.DataPreprocessor import TextPreprocessor_withStem
import pickle
lr_model = pickle.load(open(r'Shallow_ML_Models/LR_model_withStem.pkl', 'rb'))

import torch
from LSTM_Model.utils import labelencoder, predict_unseen_test
from LSTM_Model.load_data import Test_DataProcessor, create_data_loader
from LSTM_Model.LSTM_model import BERTbiLSTM
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model_weight = 'ProsusAI/finbert'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_weight)    #tokenizer to use
bert_model = BertModel.from_pretrained(bert_model_weight)    #loading base bert model

HIDDEN_DIM = 256    ## Model params for FinBERT-biLSTM_2.pt for FinBERT-biLSTM.pt change HIDDEN_DIM TO 128 and N_LAYERS TO 1
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstm_model = BERTbiLSTM(bert_model, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
lstm_model.load_state_dict(torch.load(r'LSTM_Model/FinBERT-biLSTM_2.pt', map_location=torch.device('cpu')))
lstm_model.to(device)

BATCH_SIZE = 32
MAX_LEN = 32

year12 = []
period12 = []

i = 2013
for j in range(3, 12):
    year12.append(i)
    period12.append(j+1)

for i in range(2014, 2021):
    for j in range(0, 12):
        year12.append(i)
        period12.append(j+1)

i = 2021
for j in range(0, 9):
    year12.append(i)
    period12.append(j+1)

column_name = ['positive', 'neutral', 'negative', 'exception']
sentiment_score_lr = pd.DataFrame(np.zeros((len(year12), len(column_name))), index = [year12, period12], columns = column_name).reset_index().rename(columns={"level_0": "year", "level_1": "month"}).set_index(['year','month'])
sentiment_score_lstm = pd.DataFrame(np.zeros((len(year12), len(column_name))), index = [year12, period12], columns = column_name).reset_index().rename(columns={"level_0": "year", "level_1": "month"}).set_index(['year','month'])

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys, time
from interruptingcow import timeout 

READ_FREQ = 20
SAVING_FREQ = 30

gdelt_list = pd.read_csv(r"Data/GDELT/gdelt.csv", encoding= 'unicode_escape').set_index('filename')
gdelt_colName = pd.read_csv(r"Data/GDELT/CSV.header.dailyupdates.txt", sep='\t', header=None).iloc[0].to_list()
sentiment_list = ["positive", "neutral", "negative"]
start_time = time.time()

for gdelt_list_counter in range(0, len(gdelt_list)):
    
    zip_file_url = urlopen(gdelt_list.iloc[gdelt_list_counter]['hyperlink'])
    zip_file = ZipFile(BytesIO(zip_file_url.read()))
    document_list = pd.read_csv(zip_file.open(zip_file.namelist()[0]), sep ='\t', header=None, names=gdelt_colName).set_index(['GLOBALEVENTID'])
    US_document_list = document_list[(document_list['Actor1Geo_ADM1Code'] == "US") & (document_list['Actor2Geo_ADM1Code'] == "US") & (document_list['ActionGeo_ADM1Code'] == "US") & ((document_list['Actor1Code'] == "USA") | (document_list['Actor2Code'] == "USA"))]
    text_matrix = pd.DataFrame()
    
    for doc_count in range(0,len(US_document_list)):
        
        if (doc_count % READ_FREQ == 0):
            news_year = int(str(US_document_list.iloc[doc_count, 0])[0:4])
            news_month = int(str(US_document_list.iloc[doc_count, 0])[4:6])
            news_date = int(str(US_document_list.iloc[doc_count, 0])[6:8])
            news_url = document_list.iloc[doc_count, -1]
            exception_count = 0
            
            try:
                with timeout(60, exception = RuntimeError):
                    try:
                        soup = BeautifulSoup(urlopen(news_url).read(), features="html.parser")
                        for script in soup(["script", "style"]):    # kill all script and style elements
                            script.decompose()
                        text = soup.get_text()    # get text
                        lines = (line.strip() for line in text.splitlines())    # break into lines and remove leading and trailing space on each
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))    # break multi-headlines into a line each
                        text = '\n'.join(chunk for chunk in chunks if chunk)    # drop blank lines
                        text_matrix = text_matrix.append({'sentence': text}, ignore_index=True)
                    except:
                        exception_count += 1
            except RuntimeError: 
                exception_count += 1
        
        current_time = time.time()
        sys.stdout.write("\rgdelt_list_counter = %s / %s ; doc_count = %s / %s ; time elasped: %s minutes." % (gdelt_list_counter + 1, len(gdelt_list), doc_count + 1, len(US_document_list), round((current_time-start_time)/60,2)))
        sys.stdout.flush()
    
    text_matrix['sentiment_lr'] = lr_model.predict(text_matrix['sentence'])
    text_matrix['sentiment_lstm']  = predict_unseen_test(lstm_model, create_data_loader(text_matrix, bert_tokenizer, MAX_LEN, BATCH_SIZE, True), device)['prediction']
    
    sentiment_score_lr.loc[(news_year, news_month), 'exception'] += exception_count
    sentiment_score_lstm.loc[(news_year, news_month), 'exception'] += exception_count
    for i in sentiment_list:
        sentiment_score_lr.loc[(news_year, news_month), i] += sum(text_matrix['sentiment_lr'] == i)
        sentiment_score_lstm.loc[(news_year, news_month), i] += sum(text_matrix['sentiment_lstm'] == sentiment_list.index(i))
    
    print("\rReading GEDLT progress: %s / %s ; Time Elasped: %s minutes." % (gdelt_list_counter + 1, len(gdelt_list), round((current_time-start_time)/60,2)))
    if (gdelt_list_counter % SAVING_FREQ == 0):
        sentiment_score_lr.to_csv(r"Data/sentiment_score_lr.csv")
        sentiment_score_lstm.to_csv(r"Data/sentiment_score_lstm.csv")

sentiment_score_lr.to_csv(r"Data/sentiment_score_lr.csv")
sentiment_score_lstm.to_csv(r"Data/sentiment_score_lstm.csv")
