def create_model():
    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_jobs=-1, verbose=1, class_weight='balanced')
    return rf_clf

def train():
    from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
    from misc.file_imports import get_train_data

    (X_train, X_test, y_train, y_test), class_weights = get_train_data(smote=False).values()

    rf_clf = create_model()

    rf_clf.fit(X_train, y_train)

    rf_y_pred = rf_clf.predict(X_test)
    rf_y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

    recall_rf = recall_score(y_test, rf_y_pred)
    precision_rf = precision_score(y_test, rf_y_pred)
    roc_auc_rf = roc_auc_score(y_test, rf_y_pred_proba)
    conf_matrix_rf = confusion_matrix(y_test, rf_y_pred)

    # Print metrics
    print("Recall:", recall_rf)
    print("Precision:", precision_rf)
    print("Roc-AUC:", roc_auc_rf)
    print("Confusion Matrix:\n", conf_matrix_rf)

    return rf_clf

if __name__ == '__main__':
    model = train()

    if input('\nDump? (Y for yes)') == 'Y':
        from joblib import dump
        dump(model, 'RandomForest.joblib')
