import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import joblib
import os

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

def load_dataframe(filepath):
    return pd.read_csv(filepath)

def detect_problem_type(df, target_col):
    if target_col is None or target_col == "None":
        return "Clustering"
    
    y = df[target_col]

    if y.dtype == "object" or len(y.unique()) <= 20:  
        return "Classification"
    else:
            return "Regression"
        

def split(df, target_col):

    if target_col is None or target_col == "None":
        return df, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def encoding_features(X):

    categorical_cols = X.select_dtypes(include="object").columns
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_categorical = ohe.fit_transform(X[categorical_cols])

    numeric_cols = X.select_dtypes(exclude="object").columns
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X[numeric_cols])

    X_processed = np.hstack([X_categorical, X_numeric])

    joblib.dump(ohe, os.path.join("artifacts", "onehot_encoder.pkl"))
    joblib.dump(scaler, os.path.join("artifacts", "standard_scaler.pkl"))

    return X_processed
    

def encoding_target(y):
    if y is None:
        return None
    
    if y.dtype == "object":
        lbl = LabelEncoder()
        y_encoded = lbl.fit_transform(y)   
        joblib.dump(lbl, os.path.join("artifacts", "label_encoder.pkl")) 
        return y_encoded
    else:
        return y

