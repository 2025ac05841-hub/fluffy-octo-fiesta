# =================================================================
# ML Assignment 2 : Streamlit App
#
# The app takes a csv file (test data) as input, the user can then
# select any of the 5 trained models from the dropdown and see the
# evaluation metrics, confusion matrix and classification report.
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    classification_report,
)

# all the 30 feature names of the dataset, needed to validate the csv
feature_cols = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

# these two models were trained on scaled data, so the uploaded
# data has to be scaled for them as well
scaled_models = ["Logistic Regression", "kNN"]


# training all the models once and caching them, so that the models
# are not retrained every time the app is interacted with
@st.cache_resource
def train_all_models():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100,
                                                random_state=42),
    }

    for name, model in models.items():
        if name in scaled_models:
            X_tr = scaler.transform(X_train)
            model.fit(X_tr, y_train)
        else:
            model.fit(X_train, y_train)

    return models, scaler


st.set_page_config(page_title="ML Assignment 2 - Breast Cancer App",
                   layout="wide")

st.title("Breast Cancer Wisconsin (Diagnostic)")
st.subheader("ML Assignment - 2 | M.Tech (AIML/DSE)")

st.markdown(
    "Upload the test data csv file and select a model from the dropdown "
    "to see how it performs on the test data."
)

models, scaler = train_all_models()

# ---------- sidebar ----------
with st.sidebar:
    st.header("Options")

    uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")

    selected_model = st.selectbox("Select ML Model", list(models.keys()))

    st.markdown("---")
    st.markdown("**Class labels**")
    st.markdown("- 0 : Malignant")
    st.markdown("- 1 : Benign")

    st.markdown("---")
    st.markdown(
        "[Dataset source (UCI)](https://archive.ics.uci.edu/dataset/17/"
        "breast+cancer+wisconsin+diagnostic)"
    )

# ---------- main area ----------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # check if all the required columns are present in the csv
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error("Missing columns in the uploaded file : " + ", ".join(missing_cols))
        st.stop()

    with st.expander("Preview of uploaded data"):
        st.dataframe(df.head(10))

    X = df[feature_cols]

    # scaling only for the models that need it
    if selected_model in scaled_models:
        X_model = scaler.transform(X)
    else:
        X_model = X.values

    model = models[selected_model]
    y_pred = model.predict(X_model)
    y_prob = model.predict_proba(X_model)[:, 1]

    st.subheader(f"Results : {selected_model}")

    # predictions table
    result_df = df.copy()
    result_df["Prediction"] = np.where(y_pred == 1, "Benign", "Malignant")
    st.dataframe(result_df.head(25))

    # if the csv has the diagnosis column then we can calculate metrics
    if "diagnosis" in df.columns:
        y_true = df["diagnosis"]

        st.markdown("### Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
        c2.metric("AUC Score", round(roc_auc_score(y_true, y_prob), 4))
        c3.metric("Precision", round(precision_score(y_true, y_pred), 4))

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", round(recall_score(y_true, y_pred), 4))
        c5.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
        c6.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 4))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(cm, cmap="Blues")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Malignant", "Benign"])
            ax.set_yticklabels(["Malignant", "Benign"])

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            # printing the numbers inside the matrix cells
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("### Classification Report")
            report = classification_report(
                y_true, y_pred, target_names=["Malignant", "Benign"]
            )
            st.text(report)

        st.download_button(
            "Download predictions as CSV",
            result_df.to_csv(index=False),
            file_name=f"{selected_model}_predictions.csv",
            mime="text/csv",
        )

    else:
        st.info("The uploaded file does not have a 'diagnosis' column, "
                "so metrics cannot be computed. Only predictions are shown.")

else:
    st.info(
        "Please upload the test data csv file to get started.\n\n"
        "Tip : Run train_models.py once locally, it will generate the "
        "test_data.csv file in the project folder."
    )
