import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_map = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            self.freq_map[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.freq_map[col])
        return X


#functions

def assign_rating(points):
    if points >= 95:
        return 5
    elif points >= 80:
        return 4
    elif points >= 60:
        return 3
    elif points >= 40:
        return 2
    else:
        return 1


def pipeline_transformer(data):
    # Load the pipeline from the file
    pipe = joblib.load('./model_files/preprocess_pipeline.pkl')
    prepared_data=pipe.transform(data)
    return prepared_data   


def preprocess(df):
    df.drop('user_name', axis=1, inplace=True)
    df.drop('review_title', axis=1, inplace=True)
    df.drop('region_2', axis=1, inplace=True)

    df['province']=df['country']

    keywords = ['tropical', 'fruity', 'dry', 'acid', 'ripe', 'tanni', 'berry', 'citrus', 'chocolate', 'cherry', 'lemon', 'oak', 'white', 'red', 'vanilla', 'herb', 'sweet', 'apple']
    for keyword in keywords:
        df[keyword] = df['review_description'].apply(lambda x: 1 if keyword in x else 0)
    df['sweet'] = df['review_description'].apply(lambda x: 0 if "not sweet" in x else 1 if "sweet" in x else 0)
    df['apple'] = df['review_description'].apply(lambda x: 0 if ("pineapple" in x or "custard apple" in x) else 1 if "apple" in x else 0)
    df['acid'] = df['review_description'].apply(lambda x: 0 if "not acid" in x else 1 if "acid" in x else 0)
    df['dry'] = df['review_description'].apply(lambda x: 0 if "not dry" in x else 1 if "dry" in x else 0)
    df['ripe'] = df['review_description'].apply(lambda x: 0 if "not ripe" in x else 1 if "ripe" in x else 0)    

    df.drop('review_description', axis=1, inplace=True)

    df['rating'] = df['points'].apply(assign_rating)
    
    prepared_data = pipeline_transformer(df)
    return prepared_data



def predict_variety(config,model):
    if type(config) is dict:
        df = pd.DataFrame(config)
    else:
        df = config

    preproc_data= preprocess(df)    
    print(len(preproc_data[0]))
    y_pred = model.predict(preproc_data)
    return y_pred