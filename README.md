# FinalProject

The main goal of this project to investigate if sentiment from newspaper can be useful in predicting macroeocnomics variables (unemployment rate, GDP, and CPI).

There are three steps in running the result:

1. Data Creation and Model Training:    project_create_data.ipynb;    Shallow_ML_Models/Shallow_ML_Models.ipynb;    LSTM_Model/LSTM_model.py
   
   It will first create the data, and then traing the shallow ML model and also the Bi-LSTM model
   

2. Sentiment Scoring:    project_sentiment_scoring.ipynb   OR   project_sentiment_scoring.py
   
   This will count the number of article that have a positive/neutral/negative sentiment predicted by each model

3. Macroeconomics Forecasting:    project_macro_model.ipynb
   
   This will fit the macroeconomics variables with different model: the benchmark, and with various sentiment scores.

