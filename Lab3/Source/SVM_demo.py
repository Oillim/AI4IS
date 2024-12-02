import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, classification_report
import joblib

# Load data and preprocess
data = pd.read_csv('../Data/demo_data.csv')
_data = data.copy()

columns_train = [
    'SUM_RECEIPT_AMT', 
    'COUNT_TERM_RECEIPT', 
    'TERM',
    'LA',
    'PRODUCT_NAME',
    'BAD_DEBT',
    'RESULT_YES/NO'
]

drop_columns = list(set(data.columns.to_list()) - set(columns_train))
_data = _data.drop(columns=drop_columns)

_data['BAD_DEBT'] = _data[['COUNT_TERM_RECEIPT', 'SUM_RECEIPT_AMT']].isna().all(axis=1).astype(int).astype('category')
_data['COUNT_TERM_RECEIPT'] = _data['COUNT_TERM_RECEIPT'].fillna(0)
_data['SUM_RECEIPT_AMT'] = _data['SUM_RECEIPT_AMT'].fillna(0)

mean_term = 13.35866574965612
mean_la = 14242970.936038515

oc_svm = joblib.load('oc_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
freq_encoding_product_name = joblib.load('target_mean_product_name.joblib')

_data['TERM'] = _data['TERM'].fillna(mean_term)
_data['LA'] = _data['LA'].fillna(mean_la)
_data['PRODUCT_NAME'] = _data['PRODUCT_NAME'].map(freq_encoding_product_name).fillna(0)

numeric_features = ['LA', 'TERM', 'COUNT_TERM_RECEIPT', 'SUM_RECEIPT_AMT']
_data[numeric_features] = scaler.transform(_data[numeric_features])

X_test = _data.drop(columns=['RESULT_YES/NO'])
y_test = _data['RESULT_YES/NO']

decision_scores = oc_svm.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, decision_scores)

epsilon = 1e-10
f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

auc_score = auc(recall, precision)
y_pred_new = (decision_scores >= optimal_threshold).astype(int)
average_precision = average_precision_score(y_test, y_pred_new)
report = classification_report(y_test, y_pred_new)

print(f'Precision-Recall AUC: {auc_score}')
print(report)
print(f"Average Precision: {average_precision}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
