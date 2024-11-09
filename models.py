from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# Define search space for HyperOpt
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
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth']) if params['max_depth'] > 0 else -1
    params['min_child_samples'] = int(params['min_child_samples'])
    params['n_estimators'] = int(params['n_estimators'])
    clf = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt', class_weight='balanced', **params)
    
    score = cross_val_score(clf, X, y, scoring='roc_auc', cv=5).mean()
    return {'loss': -score, 'status': STATUS_OK}

def tune_model(X, y, max_evals=50):
    trials = Trials()
    best = fmin(fn=lambda params: objective(params, X, y), space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best
