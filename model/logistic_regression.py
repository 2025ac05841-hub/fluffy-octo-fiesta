# Logistic regression on the breast cancer dataset.
# Run this file directly to train the model and see how it performs.

import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# sklearn ships the breast cancer dataset with itself, no need to download anything
data = load_breast_cancer()
X, y = data.data, data.target

# hold out 20% of the data for testing, stratified so both classes stay balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# the features have very different ranges (some are tiny, others go up to 3000),
# so we standardize them first — without this the model barely converges
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# bumped max_iter to 1000 because the default sometimes throws convergence warnings
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("===== Logistic Regression =====")
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
print("AUC       :", round(roc_auc_score(y_test, y_prob), 4))
print("Precision :", round(precision_score(y_test, y_pred), 4))
print("Recall    :", round(recall_score(y_test, y_pred), 4))
print("F1 score  :", round(f1_score(y_test, y_pred), 4))
print("MCC       :", round(matthews_corrcoef(y_test, y_pred), 4))

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "logistic_regression.pkl")
joblib.dump(model, model_path)
print("Model saved at", model_path)
