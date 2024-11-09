import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
from misc.file_imports import get_train_data
from joblib import dump

# Определение пространства поиска гиперпараметров
space = {
    'iterations': hp.quniform('iterations', 100, 1000, 50),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'depth': hp.quniform('depth', 3, 10, 1),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'border_count': hp.quniform('border_count', 32, 255, 1),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    'random_strength': hp.uniform('random_strength', 1, 10),
    'od_type': hp.choice('od_type', ['IncToDec', 'Iter']),
    'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 10, 1),
}

def objective(params, X, y, class_weights, X_val, y_val):
    # Преобразование параметров в целые числа там, где это необходимо
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    params['border_count'] = int(params['border_count'])
    params['leaf_estimation_iterations'] = int(params['leaf_estimation_iterations'])

    clf = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        class_weights=class_weights,
        random_seed=42,
        verbose=0,
        **params
    )

    # Используем кросс-валидацию для оценки модели
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            clf, X, y,
            cv=cv,
            scoring='roc_auc',
            fit_params={'early_stopping_rounds': 50, 'eval_set': [(X_val, y_val)]}
        )
        roc_auc = scores.mean()
        loss = -roc_auc
        status = STATUS_OK
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        loss = np.inf
        status = STATUS_OK

    return {'loss': loss, 'status': status}

def create_model(best_params, class_weights):
    # Создаем модель с лучшими параметрами
    cat_clf = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
        **best_params
    )
    return cat_clf

def train():
    (X_train, X_test, y_train, y_test), class_weights = get_train_data(smote=False).values()

    # Запуск оптимизации
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X_train, y_train, class_weights, X_test, y_test),
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )

    print("Лучшие параметры:", best)

    # Преобразование параметров после оптимизации
    best_params = {
        'iterations': int(best['iterations']),
        'learning_rate': best['learning_rate'],
        'depth': int(best['depth']),
        'l2_leaf_reg': best['l2_leaf_reg'],
        'border_count': int(best['border_count']),
        'bagging_temperature': best['bagging_temperature'],
        'random_strength': best['random_strength'],
        'od_type': ['IncToDec', 'Iter'][best['od_type']],
        'leaf_estimation_iterations': int(best['leaf_estimation_iterations'])
    }

    cat_clf = create_model(best_params, class_weights)
    cat_clf.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    y_pred = cat_clf.predict(X_test)
    y_pred_proba = cat_clf.predict_proba(X_test)[:, 1]

    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return cat_clf

if __name__ == '__main__':
    model = train()

    if input('\nDump? (Y for yes)') == 'Y':
        dump(model, 'Catboost.joblib')
