import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import compute_class_weight

def get_train_data(smote=False):
    file_path = "../../data/train.csv"
    df = pd.read_csv(file_path)
    df.head()

    X = df.drop(["target"],axis=1)
    y = df["target"]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE()

        X_train, y_train = smote.fit_resample(X_train, y_train)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

    return {'data': (X_train, X_test, y_train, y_test), 'weights': class_weights_dict}

def get_test_data():
    file_path = "../data/test.csv"
    df = pd.read_csv(file_path)
    df.head()

    scaler = StandardScaler()
    scaler.fit(df)
    output_X = scaler.transform(df)

    output_df = pd.read_csv('../data/input/baseline_submission_case1.csv', index_col='id')

    return output_X, output_df
