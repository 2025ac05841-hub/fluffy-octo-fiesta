# Machine Learning Assignment - 2

## M.Tech (AIML/DSE)

### Classification Models and Streamlit Deployment

---

## 1. Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification algorithms using a common classification dataset.

The implemented models are evaluated using multiple classification metrics and deployed through an interactive Streamlit web application.

The project demonstrates an end-to-end machine learning workflow involving:

- Dataset preparation
- Exploratory data understanding
- Data preprocessing
- Model training
- Model evaluation
- Comparison of classification models
- Interactive Streamlit application development
- Model prediction using test data
- Deployment on Streamlit Community Cloud

---

## 2. Dataset Description

### Dataset Name

**Breast Cancer Wisconsin (Diagnostic) Dataset**

### Dataset Source

UCI Machine Learning Repository

### Problem Type

Binary Classification

### Objective

The objective is to classify a breast tumor as either:

- **Malignant**
- **Benign**

based on numerical measurements computed from digitized images of breast mass cell nuclei.

### Dataset Size

| Property | Value |
|---|---:|
| Number of Instances | 569 |
| Number of Features | 30 |
| Number of Classes | 2 |
| Classification Type | Binary |
| Missing Values | None |

The dataset satisfies the assignment requirements of a minimum of **500 instances** and **12 features**.

### Target Variable

The target variable contains two classes:

- `M` - Malignant
- `B` - Benign

For machine learning implementation, the target is encoded numerically (0 = Malignant, 1 = Benign).

---

## 3. GitHub Repository

The complete source code, trained model files, test data, requirements file, and README are available in the GitHub repository.

**GitHub Repository:**  
`<YOUR_GITHUB_REPOSITORY_LINK>`

### Repository Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── random_forest.py
│
└── screenshots/
    └── bits_lab_execution.png
```

The `model/` directory contains the **model source files (`*.py`)** for all
implemented models. The Streamlit app imports and trains these models
directly at startup, so no serialized model files are stored in the
repository. Running `train_models.py` locally saves the trained models as
`.pkl` files (ignored via `.gitignore`).

---

## 4. Machine Learning Models

The following classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

All models were evaluated using the same test dataset (`test_data.csv`, a 20% stratified split with 114 instances).

---

## 5. Evaluation Metrics

The following evaluation metrics were calculated for each classification model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Metric Description

#### Accuracy

Accuracy represents the proportion of correctly classified observations among all observations.

```text
Accuracy = Correct Predictions / Total Predictions
```

#### AUC Score

AUC represents the model's ability to distinguish between the two classes across different classification thresholds.

A higher AUC indicates better class discrimination.

#### Precision

Precision measures the proportion of predicted positive observations that are actually positive.

```text
Precision = TP / (TP + FP)
```

#### Recall

Recall measures the proportion of actual positive observations that are correctly identified.

```text
Recall = TP / (TP + FN)
```

#### F1 Score

F1 Score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Matthews Correlation Coefficient

MCC measures the quality of binary classifications by considering true positives, true negatives, false positives, and false negatives.

The MCC value ranges from:

```text
-1 to +1
```

A value close to +1 indicates a strong prediction, while a value close to 0 indicates performance close to random prediction.

---

## 6. Model Comparison

The following table presents the evaluation results obtained on the test data (114 instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

---

## 7. Observations on Model Performance

### Logistic Regression

Logistic Regression provides a baseline linear classification model. Its performance indicates how effectively the target classes can be separated using a linear decision boundary.

**Observation:**  
Logistic Regression was the best performing model with the highest accuracy (0.9825), AUC (0.9954), precision, recall, F1 (all 0.9861) and MCC (0.9623). It misclassified only 2 of 114 test instances. The Wisconsin dataset features are strongly linearly separable, so a linear boundary is sufficient to separate malignant from benign tumors.

---

### Decision Tree

The Decision Tree classifier uses a sequence of feature-based decisions to classify observations. It can capture non-linear relationships between the features and target variable.

**Observation:**  
The Decision Tree had the weakest performance overall (accuracy 0.9123, AUC 0.9157, MCC 0.8174). It lost performance primarily on recall (0.9028), misclassifying the most malignant samples (7 false negatives). This suggests the single tree overfits training data and is sensitive to small variations in the feature space.

---

### K-Nearest Neighbor

KNN classifies an observation based on the classes of its nearest neighboring observations. Feature scaling is important for KNN because the algorithm is distance-based.

**Observation:**  
kNN performed well (accuracy 0.9561, F1 0.9655, MCC 0.9054) with the highest recall along with Random Forest (0.9722). It benefited from standard scaling, which prevented high-magnitude features such as mean area from dominating the distance computation. kNN tied for second place overall.

---

### Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm that assumes conditional independence between features.

**Observation:**  
Naive Bayes delivered good results (accuracy 0.9386, AUC 0.9878, F1 0.9517, MCC 0.8676). Its probability-based ranking is strong (second-best AUC), even though the conditional independence assumption is violated by the highly correlated features of this dataset. It performed slightly weaker than the linear and distance-based models.

---

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to produce a final prediction. It can generally model complex non-linear relationships.

**Observation:**  
Random Forest was the best ensemble model and second overall (accuracy 0.9561, AUC 0.9937, F1 0.9655, MCC 0.9054). Bagging across many trees reduced the variance seen in the single decision tree, improving recall substantially (0.9028 → 0.9722). Its near-perfect AUC (0.9937) shows excellent class discrimination.

---

## 8. Overall Winner

Based on the evaluation metrics, the model with the best overall performance is:

### Overall Winner

**Logistic Regression**

with Accuracy 0.9825, AUC 0.9954, Precision 0.9861, Recall 0.9861, F1 0.9861 and MCC 0.9623.

It ranked first on all six metrics. The dataset's high separability makes a regularized linear model the strongest and most stable choice, followed closely by Random Forest and kNN.

---

## 9. Streamlit Application

An interactive Streamlit application has been developed to demonstrate the trained classification models.

### Streamlit Application Features

The application provides the following functionality:

- CSV test-data upload
- Model selection using a dropdown
- Model evaluation
- Display of evaluation metrics
- Confusion matrix
- Classification report
- Prediction results on uploaded test data
- Download of prediction results (CSV)

The application allows the user to select a classification model and observe its performance on the uploaded test dataset.

---

## 10. Streamlit Application Link

**Live Streamlit Application:**

`<YOUR_STREAMLIT_APP_LINK>`

The application is deployed using Streamlit Community Cloud.

---

## 11. Application Workflow

The application follows the workflow below:

```text
User uploads test CSV
        |
        v
Data validation
        |
        v
Feature preprocessing
        |
        v
User selects ML model
        |
        v
Selected trained model
        |
        v
Prediction
        |
        +--------------------+
        |                    |
        v                    v
Evaluation Metrics     Confusion Matrix
        |
        v
Classification Report
```

---

## 12. Test Data

The repository contains the test dataset used for evaluating the trained models.

File:

```text
test_data.csv
```

It holds 114 instances (20% stratified split) with the 30 features and the `diagnosis` ground-truth column. It is generated by running `train_models.py`.

The Streamlit application accepts this CSV file through the dataset upload option.

Only test data is uploaded through the application to keep the deployed application lightweight.

---

## 13. Requirements

The project dependencies are specified in `requirements.txt`.

Main libraries used:

```text
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
```

---

## 14. Running the Project Locally

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
```

### Step 2: Navigate to the Project Directory

```bash
cd ML_Assignment_2
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Models (Optional, models are already saved)

```bash
python train_models.py
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The Streamlit application will then open in the browser.

---

## 15. Model Training

The models were trained using the selected classification dataset.

The training workflow includes:

1. Loading the dataset
2. Data preprocessing
3. Encoding the target variable
4. Splitting the dataset into training and testing datasets (80/20 stratified)
5. Feature scaling where required
6. Training the classification models (defined in `model/*.py`)
7. Generating predictions
8. Calculating evaluation metrics
9. Saving test data (`test_data.csv`) and the trained models

The model definitions are maintained as `*.py` source files in the `model/`
directory. Running `train_models.py` saves the trained models to disk, while
the Streamlit application trains them at startup directly from the source
files.

---

## 16. BITS Virtual Lab Execution

The assignment was executed using the BITS Virtual Lab as required.

A screenshot showing the assignment execution in the BITS Virtual Lab is included in the project/submission.

**Screenshot:**

`BITs Virtual Lab execution screenshot` (see `screenshots/bits_lab_execution.png`)

---

## 17. Conclusion

This project implements and compares multiple classification algorithms on the Breast Cancer Wisconsin Diagnostic dataset.

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient. Logistic Regression was identified as the overall winner, with Random Forest and kNN close behind.

An interactive Streamlit application was also developed to allow users to upload test data, select a trained classification model, view evaluation metrics, and analyze the model predictions using a confusion matrix and classification report.

The project demonstrates the complete workflow from machine learning model development and evaluation to interactive application development and cloud deployment.

---

## 18. Author

**Name:** `<YOUR NAME>`  
**Programme:** M.Tech (AIML/DSE)  
**Course:** Machine Learning  
**Assignment:** Assignment - 2

---

## 19. Links

| Resource | Link |
|---|---|
| GitHub Repository | `<YOUR_GITHUB_REPOSITORY_LINK>` |
| Live Streamlit Application | `<YOUR_STREAMLIT_APP_LINK>` |
| Dataset Source | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |