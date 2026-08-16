# ------------------------------------------------------------
# ML Assignment 2 : Training script
# Trains all the 5 classification models on the breast cancer
# dataset and prints the comparison table with all the metrics
# ------------------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ------------------------------------------------------------
# 1. Loading the dataset
# ------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target  # 0 = Malignant, 1 = Benign

print("Dataset shape :", df.shape)
print("Class distribution :")
print(df["diagnosis"].value_counts())
print()

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 80-20 split, stratify is used so that the ratio of classes
# remains the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size :", X_train.shape)
print("Test size  :", X_test.shape)
print()

# saving the test data, this csv will be uploaded in the streamlit app
test_df = X_test.copy()
test_df["diagnosis"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("test_data.csv saved")
print()

# scaling the features. Needed for logistic regression and knn
# because both of them are distance/gradient based
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 2. Training the models one by one
# ------------------------------------------------------------

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
print("Logistic Regression done")

# --- Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
print("Decision Tree done")

# --- kNN (k = 5) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
print("kNN done")

# --- Gaussian Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:, 1]
print("Naive Bayes done")

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print("Random Forest done")
print()

# ------------------------------------------------------------
# 3. Calculating all the metrics
# ------------------------------------------------------------
model_names = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
]

y_preds = [y_pred_lr, y_pred_dt, y_pred_knn, y_pred_nb, y_pred_rf]
y_probs = [y_prob_lr, y_prob_dt, y_prob_knn, y_prob_nb, y_prob_rf]

results = []
for name, y_pred, y_prob in zip(model_names, y_preds, y_probs):
    results.append(
        {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
    )
    # confusion matrix for each model, will be useful for the report
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} :")
    print(cm)
    print()

comparison = pd.DataFrame(results)
pd.set_option("display.float_format", "{:.4f}".format)
print()
print("=========== MODEL COMPARISON TABLE ===========")
print(comparison.to_string(index=False))

# ------------------------------------------------------------
# 4. Plotting the confusion matrices together
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for i, name in enumerate(model_names):
    cm = confusion_matrix(y_test, y_preds[i])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[i],
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
    )
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print()
print("confusion_matrices.png saved")

# ------------------------------------------------------------
# 5. Saving the trained models
# ------------------------------------------------------------
models = [lr, dt, knn, nb, rf]
model_dir = os.path.join(os.path.dirname(__file__), "model")
for name, model in zip(model_names, models):
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(model_dir, filename))

joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
print("All models saved in the model/ folder")
