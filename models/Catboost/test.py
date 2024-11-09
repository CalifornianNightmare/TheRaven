from misc.file_imports import get_test_data
from joblib import load

model = load('RandomForest.joblib')

X, out_df = get_test_data()

output = model.predict_proba(X)[:,1]
out_df['target'] = output
out_df.to_csv('../data/output/RandomForest_submission.csv', index=True)
