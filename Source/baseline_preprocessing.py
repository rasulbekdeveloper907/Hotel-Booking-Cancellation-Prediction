# Source/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Cleaner:
    def __init__(self):
        self.fill_values = {}

    def fit(self, X):
        for col in X.columns:
            if X[col].dtype == 'object':
                self.fill_values[col] = X[col].mode()[0]
            else:
                self.fill_values[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.fill_values.items():
            X[col] = X[col].fillna(value)
        return X

class Encoder:
    def __init__(self, max_unique=5):
        self.max_unique = max_unique
        self.cat_cols = None
        self.dummy_cols = {}

    def fit(self, X):
        self.cat_cols = X.select_dtypes(include='object').columns
        for col in self.cat_cols:
            if X[col].nunique() <= self.max_unique:
                self.dummy_cols[col] = X[col].unique().tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols:
            if col in self.dummy_cols:
                for val in self.dummy_cols[col]:
                    X[f"{col}_{val}"] = (X[col] == val).astype(int)
                X.drop(columns=[col], inplace=True)
            else:
                X[col] = X[col].astype('category').cat.codes
        return X

class Scaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_cols = None

    def fit(self, X):
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X
