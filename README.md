# Classification Model Comparison

## Problem Statement
This project compares five machine learning classification algorithms on the Breast Cancer Wisconsin Diagnostic Dataset.

The objective is to evaluate their performance using six classification metrics and provide an interactive Streamlit application for testing saved models.

## Dataset Description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Total instances: 569
- Input features: 30 numerical features
- Target classes: 0 = Malignant, 1 = Benign
- Training records: 455
- Testing records: 114
- Train-test split: 80:20 (stratified)
- Missing values: 0

Dataset URL:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Classification Algorithms
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics
- Accuracy
- Area Under ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Precision, recall, and F1 use class 1 (Benign) as the positive class.

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## Performance Observations
- Logistic Regression: Highest AUC (0.9954), with excellent overall classification performance.
- Decision Tree: Lowest accuracy and MCC among the five models.
- KNN: Strong performance with 0.9561 accuracy and 0.9655 F1 score.
- Naive Bayes: High AUC (0.9878), despite lower accuracy than KNN.
- SVM: Tied with Logistic Regression on five metrics, with a slightly lower AUC.

## Best Performing Model
Logistic Regression achieved the highest AUC of 0.9954.

Logistic Regression and SVM both achieved 98.25% accuracy.

## Streamlit Application
The application allows users to:
- Upload a CSV file containing the required features and target.
- Select one of five saved classification models.
- Generate predictions without retraining.
- View six evaluation metrics.
- View a confusion matrix.

## Running the Application
Install dependencies:

    pip install -r requirements.txt

Start Streamlit:

    streamlit run app.py

Upload test_data.csv to evaluate the models.

## Project Links
GitHub Repository: Add URL after uploading project.

Deployed Streamlit App: Add URL after deployment.
