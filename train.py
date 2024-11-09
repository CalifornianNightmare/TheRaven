from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb

def train_lgb(X_train, y_train, params):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    results = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return results
