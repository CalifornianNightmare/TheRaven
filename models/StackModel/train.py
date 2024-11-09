from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix

class EstimatorWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, name, **kwargs):
        self.name = name
        self.base_estimator = base_estimator
        self.base_estimator.set_params(**kwargs)

    def fit(self, X, y, **fit_params):
        print(f"Начинается обучение модели {self.name}")
        self.base_estimator.fit(X, y, **fit_params)
        self.classes_ = self.base_estimator.classes_
        print(f"Обучение модели {self.name} завершено")
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = {'name': self.name, 'base_estimator': self.base_estimator}
        if deep:
            base_params = self.base_estimator.get_params(deep=deep)
            params.update(base_params)
        return params

    def set_params(self, **params):
        if 'name' in params:
            self.name = params.pop('name')
        if 'base_estimator' in params:
            self.base_estimator = params.pop('base_estimator')
        if params:
            self.base_estimator.set_params(**params)
        return self

def create_estimators(class_weights):
    from LightGBM.train import create_model as LGBM
    from Catboost.train import create_model as Catboost
    from RandomForest.train import create_model as RandomForest

    lgb_clf_base = LGBM()
    cat_clf_base = Catboost(get_train_data(smote=False)['weights'])
    rf_clf_base = RandomForest()

    lgb_clf_wrapped = EstimatorWrapper(lgb_clf_base, name='LightGBM', n_estimators=100, verbosity=2, n_jobs=-1)
    cat_clf_wrapped = EstimatorWrapper(cat_clf_base, name='CatBoost',
                                       iterations=1000,
                                       learning_rate=0.1,
                                       depth=6,
                                       eval_metric='AUC',
                                       verbose=2)
    rf_clf_wrapped = EstimatorWrapper(rf_clf_base, name='RandomForest', verbose=2, n_jobs=-1)

    return [
        ('lgb', lgb_clf_wrapped),
        ('cat', cat_clf_wrapped),
        ('rf', rf_clf_wrapped)
    ]

def create_meta_estimator():
    meta_clf_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        verbosity=2
    )
    return EstimatorWrapper(meta_clf_base, name='XGBoost')

def train_stacking_classifier(X_train, y_train, class_weights):
    estimators = create_estimators(class_weights)
    meta_clf_wrapped = create_meta_estimator()

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf_wrapped,
        cv=5,
        verbose=2
    )

    stacking_clf.fit(X_train, y_train)
    return stacking_clf

def evaluate_model(stacking_clf, X_test, y_test):
    y_pred = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]

    recall_stack = recall_score(y_test, y_pred)
    precision_stack = precision_score(y_test, y_pred)
    roc_auc_stack = roc_auc_score(y_test, y_pred_proba)
    conf_matrix_stack = confusion_matrix(y_test, y_pred)

    print("Recall:", recall_stack)
    print("Precision:", precision_stack)
    print("ROC-AUC:", roc_auc_stack)
    print("Confusion Matrix:\n", conf_matrix_stack)

if __name__ == '__main__':
    from misc.file_imports import get_train_data

    (X_train, X_test, y_train, y_test), class_weights = get_train_data(smote=True).values()
    stacking_clf = train_stacking_classifier(X_train, y_train, class_weights)
    evaluate_model(stacking_clf, X_test, y_test)

    if input('\nDump? (Y for yes)') == 'Y':
        from joblib import dump
        dump(stacking_clf, 'StackModel.joblib')
