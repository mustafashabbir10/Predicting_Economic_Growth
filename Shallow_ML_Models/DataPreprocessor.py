from sklearn.base import TransformerMixin, BaseEstimator
import nltk
from nltk.stem import PorterStemmer
import string
import re

class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self,n_jobs=-1, Stemming=False):
        self.n_jobs   = n_jobs
        self.Stemming = Stemming
    
    def fit(self, X,y):
        return self
    
    def transform(self, X):
        lower_case_text       = X.apply(lambda x:x.lower())
        removed_punct_text    = lower_case_text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        removed_numbers_text  = removed_punct_text.apply(lambda x: re.sub(" \d+", " ", x))
        clear_whitespace_text = removed_numbers_text.apply(lambda x: re.sub(' +', ' ', x.lstrip().rstrip()))
        if self.Stemming:
            clear_whitespace_text = Stemdata(clear_whitespace_text)
        return clear_whitespace_text
    
    def Stemdata(self, X):
        text_tokens   = X.apply(lambda x: x.split())
        stemed_tokens = text_tokens.apply(lambda x: [self.ps.stem(word) for word in x])
        return stemed_tokens.apply(lambda x: ' '.join(x))

    
    

class TextPreprocessor_withStem(BaseEstimator, TransformerMixin):
    
    def __init__(self,n_jobs=-1):
        self.n_jobs    = n_jobs
        self.ps        = PorterStemmer()
    
    def fit(self, X,y):
        return self
    
    def transform(self, X):
        lower_case_text       = X.apply(lambda x:x.lower())
        removed_punct_text    = lower_case_text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        removed_numbers_text  = removed_punct_text.apply(lambda x: re.sub(" \d+", " ", x))
        clear_whitespace_text = removed_numbers_text.apply(lambda x: re.sub(' +', ' ', x.lstrip().rstrip()))
        text_tokens           = clear_whitespace_text.apply(lambda x: x.split())
        remove_stopwords_stem = text_tokens.apply(lambda x: [self.ps.stem(word) for word in x])


        return remove_stopwords_stem.apply(lambda x: ' '.join(x))