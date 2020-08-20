import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy import sparse

class DataFrameConverter(TransformerMixin):
    
    def __init__(self, column_names=None):
        self.column_names=column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        if sparse.issparse(X_):
            X_ = X_.todense()
        
        if self.column_names:
            return pd.DataFrame(X_, columns=self.column_names) 
        else:
            return pd.DataFrame(X_)

class SafetyFeatureEngineering(TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        X_ = X.copy()

        X_['safety_equipment'] = X_['secu'].map(lambda x: str(round(x))[0])
        X_['is_safety_equipment'] = X_.secu.map(lambda x: str(round(x))[1])

        return X_.drop(columns=['secu'])

class HourParser(TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        X_['hour_of_day'] = X_['hrmn'].map(lambda x: int((str(x)[0:-2]).replace('', '0')))
        
        return X_.drop(columns=['hrmn'])
