import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df, target_col="target", drop_cols=["smpl"]):
    df = df.drop(columns=drop_cols)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def scale_data(X_train, X_test=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def resample_data(X, y):
    smote = SMOTE()
    return smote.fit_resample(X, y)
