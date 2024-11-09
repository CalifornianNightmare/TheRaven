import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from misc.file_imports import get_train_data
from joblib import dump

# Определение пространства поиска гиперпараметров
space = {
    'num_leaves': hp.quniform('num_leaves', 20, 100, 1),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'min_child_samples': hp.quniform('min_child_samples', 5, 50, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
}

def objective(params, X, y):
    # Преобразуем параметры, которые должны быть целыми числами
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth']) if params['max_depth'] > 0 else -1
    params['min_child_samples'] = int(params['min_child_samples'])
    params['n_estimators'] = int(params['n_estimators'])

    # Удаляем параметры с некорректными значениями
    if params['max_depth'] == 0:
        params['max_depth'] = -1

    print(f"Текущие параметры: {params}")

    clf = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        class_weight='balanced',
        **params
    )

    # Используем кросс-валидацию для оценки модели
    try:
        score = cross_val_score(
            clf, X, y,
            scoring='roc_auc',
            cv=5
        ).mean()
        loss = -score
        status = STATUS_OK
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        loss = np.inf
        status = STATUS_OK

    return {'loss': loss, 'status': status}

def create_model(best_params):
    # Создаем модель с лучшими параметрами
    lgb_clf = lgb.LGBMClassifier(**best_params)
    return lgb_clf

def train():
    X_train, X_test, y_train, y_test = get_train_data(smote=True)['data']

    # Запуск оптимизации
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X_train, y_train),
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    print("Лучшие параметры:", best)

    # Преобразуем параметры после оптимизации
    best_params = {
        'num_leaves': int(best['num_leaves']),
        'max_depth': int(best['max_depth']) if best['max_depth'] > 0 else -1,
        'learning_rate': best['learning_rate'],
        'min_child_samples': int(best['min_child_samples']),
        'n_estimators': int(best['n_estimators']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
    }

    lgb_clf = create_model(best_params)
    lgb_clf.fit(X_train, y_train)

    y_pred = lgb_clf.predict(X_test)
    y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]

    print("recall:", recall_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred))
    print("roc-auc:", roc_auc_score(y_test, y_pred_proba))
    print("confusion matrix:", confusion_matrix(y_test, y_pred))

    return lgb_clf

if __name__ == '__main__':
    model = train()

    if input('\nDump? (Y for yes)') == 'Y':
        dump(model, 'LightGBM.joblib')
