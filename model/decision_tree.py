# Decision tree on the breast cancer dataset.
# Run this file directly to train the model and see how it performs.

import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# tree based models don't care about feature scaling, so we skip the scaler here
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("===== Decision Tree =====")
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
print("AUC       :", round(roc_auc_score(y_test, y_prob), 4))
print("Precision :", round(precision_score(y_test, y_pred), 4))
print("Recall    :", round(recall_score(y_test, y_pred), 4))
print("F1 score  :", round(f1_score(y_test, y_pred), 4))
print("MCC       :", round(matthews_corrcoef(y_test, y_pred), 4))

# just curious how deep the tree ended up growing
print("Depth of the tree :", model.get_depth())

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "decision_tree.pkl")
joblib.dump(model, model_path)
print("Model saved at", model_path)
